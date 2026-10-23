"""Measure isolated smoke checks or a persistent MOSS generation session."""

import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

SCHEMA_VERSION = 2


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return importlib.import_module(name + ".nodes") if Path(path).name == "__init__.py" else module


def build_requests(config):
    first = dict(name="first", prompt=config["prompt"], seed=config["seed"],
                 steps=config["steps"], cfg=config["cfg"])
    return [first,
            dict(first, name="warm_same_prompt", seed=config["seed"] + 1),
            dict(first, name="warm_new_prompt", prompt=config["warm_prompt"]),
            dict(first, name="warm_settings", prompt=config["warm_prompt"],
                 steps=config["warm_steps"], cfg=config["warm_cfg"])]


class CandidateSession:
    def __init__(self, config):
        self.config = config
        self.started = time.monotonic()
        self.torch = None
        self.options_enabled = set(config.get("options", [config.get("option", "none")]))
        self.embedding_cache = {}
        self.spectrum = None
        self.result = dict(schema_version=SCHEMA_VERSION, candidate=config["id"],
                           mode=config.get("mode", "smoke"), status="RUNNING", stage="import",
                           timings_s={}, events=[], scenarios=[], config=config,
                           stage_started=self.started)
        self.persist()

    def persist(self):
        write_json(self.config["state_path"], self.result)

    def synchronize(self):
        if self.torch is not None and self.torch.cuda.is_initialized():
            self.torch.cuda.synchronize()

    def stage(self, name, function, **details):
        self.synchronize()
        self.result.update(stage=name, stage_started=time.monotonic())
        self.persist()
        begin = time.monotonic()
        event = dict(stage=name, **details)
        try:
            value = function()
            self.synchronize()
            event["status"] = "PASS"
            return value
        except Exception:
            event["status"] = "FAIL"
            raise
        finally:
            event["seconds"] = round(time.monotonic() - begin, 6)
            self.result["timings_s"][name] = event["seconds"]
            self.result["events"].append(event)
            self.persist()

    def initialize(self):
        root = Path(self.config["comfy_root"])
        os.chdir(root)
        sys.path.insert(0, str(root))
        sys.argv = ["moss-measure", *self.config.get("flags", [])]
        import comfy.options
        comfy.options.enable_args_parsing()
        from comfy.cli_args import args, enables_dynamic_vram
        import cuda_malloc
        import comfy_aimdo.control as aimdo
        if enables_dynamic_vram():
            headroom = None if args.reserve_vram is None else int(args.reserve_vram * 2**30)
            aimdo.init(simple_vram_headroom=headroom, nvml_pressure=not args.disable_nvml_pressure)
        import torch
        import nodes
        import comfy.memory_management as dynamic_memory
        import comfy.model_management as memory
        import comfy.model_patcher as patchers
        import comfy.sampler_helpers as helpers
        import comfy.samplers as samplers
        self.torch, self.nodes, self.memory = torch, nodes, memory
        self.patchers, self.helpers, self.samplers = patchers, helpers, samplers
        if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
            raise RuntimeError("T4 GPU를 선택하세요.")
        if enables_dynamic_vram() and memory.torch_version_numeric >= (2, 8):
            devices = [(device.index, int(args.vram_headroom * 2**30))
                       for device in memory.get_all_torch_devices()]
            if aimdo.init_devices(iter(devices)):
                patchers.CoreModelPatcher = patchers.ModelPatcherDynamic
                dynamic_memory.aimdo_enabled = True
        self.result.update(dynamic_vram=dynamic_memory.aimdo_enabled,
                           gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda)
        torch.cuda.reset_peak_memory_stats()

    def load_models(self):
        self.load_started = time.monotonic()
        cfg = self.config
        self.model = self.stage("load_dit", lambda: self.nodes.UNETLoader().load_unet(
            cfg["dit"], cfg.get("weight_dtype", "default"))[0])
        self.clip = self.stage("load_te", lambda: self.nodes.CLIPLoader().load_clip(
            cfg["te"], "moss_image", cfg.get("text_device", "default"))[0])
        self.vae = self.stage("load_vae", lambda: self.nodes.VAELoader().load_vae(cfg["vae"])[0])
        if "lora8" in self.options_enabled:
            def patch_lora():
                patched = self.nodes.LoraLoaderModelOnly().load_lora_model_only(
                    self.model, cfg["lora"], cfg.get("lora_strength", 1.0))[0]
                self.result["lora_patch_keys"] = len(patched.patches)
                if not patched.patches:
                    raise ValueError("적용된 LoRA patch가 없습니다.")
                return patched
            self.model = self.stage("patch_lora", patch_lora)
        if "compile" in self.options_enabled:
            from comfy_extras.nodes_torch_compile import TorchCompileModel
            self.model = self.stage("patch_compile", lambda: TorchCompileModel.execute(self.model, "inductor")[0])
        if "spectrum" in self.options_enabled:
            self.spectrum = self.stage("import_spectrum", lambda: load_module(
                "moss_spectrum_probe", cfg["spectrum_path"]))
            settings = self.spectrum.SpectrumConfig(**cfg["spectrum_config"])
            self.model = self.stage("patch_spectrum", lambda: self.spectrum.apply_spectrum_moss(self.model, settings))
        self.result["load_and_patch_s"] = round(time.monotonic() - self.load_started, 6)

    def check_tensor(self, tensor, expected=None):
        if expected is not None and tuple(tensor.shape) != tuple(expected):
            raise ValueError(f"shape={tuple(tensor.shape)}, expected={tuple(expected)}")
        if not bool(self.torch.isfinite(tensor).all().item()):
            raise FloatingPointError("NaN/Inf가 발생했습니다.")

    def encode(self, text):
        if text in self.embedding_cache:
            return self.embedding_cache[text], True
        output = self.clip.encode_from_tokens(self.clip.tokenize(text), return_dict=True)
        tensor = output["cond"]
        self.check_tensor(tensor)
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[-1] != 4096 or tensor.shape[1] < 1:
            raise ValueError(f"잘못된 conditioning shape: {tuple(tensor.shape)}")
        output["cond"] = tensor.cpu()
        self.embedding_cache[text] = output
        return output, False

    def shape(self):
        return (1, 128, self.config["height"] // 16, self.config["width"] // 16)

    def spectrum_counts(self):
        if self.spectrum is None:
            return dict(actual=0, forecast=0)
        wrappers = self.model.model_options["transformer_options"].get("wrappers", {})
        roots = {id(wrapper.root_state): wrapper.root_state for group in wrappers.values()
                 for values in group.values() for wrapper in values if hasattr(wrapper, "root_state")}
        branches = [b for root in roots.values() for b in root.branches.values()]
        return dict(actual=sum(b.actual_count for b in branches), forecast=sum(b.forecast_count for b in branches))

    def decode(self, samples):
        if "tiled_vae" in self.options_enabled:
            pixels = self.nodes.VAEDecodeTiled().decode(self.vae, {"samples": samples}, 512, 64)[0]
        else:
            pixels = self.vae.decode(samples)
        self.check_tensor(pixels, (1, self.config["height"], self.config["width"], 3))
        return pixels

    def track_vae_path(self):
        self.result["vae_actual_path"] = "tiled" if "tiled_vae" in self.options_enabled else "default"
        if "tiled_vae" not in self.options_enabled:
            for method in ("decode_tiled_", "decode_tiled_1d", "decode_tiled_3d", "_decode_tiled_owned"):
                original = getattr(self.vae, method)
                def tracked(*args, _original=original, **kwargs):
                    self.result["vae_actual_path"] = "auto_tiled"
                    return _original(*args, **kwargs)
                setattr(self.vae, method, tracked)

    def smoke(self):
        torch = self.torch
        positive, _ = self.stage("smoke.encode_positive", lambda: self.encode(self.config["prompt"]))
        negative, _ = self.stage("smoke.encode_negative", lambda: self.encode(""))
        self.result.update(positive_tokens=positive["cond"].shape[1], negative_tokens=negative["cond"].shape[1])
        device = self.memory.get_torch_device()
        def setup():
            latent = torch.randn(self.shape(), generator=torch.Generator("cpu").manual_seed(self.config["seed"]))
            options = self.patchers.create_model_options_clone(self.model.model_options)
            self.helpers.prepare_model_patcher(self.model, {}, options)
            return latent, options
        latent, options = self.stage("smoke.prepare_inputs", setup)
        def prepare():
            self.memory.load_models_gpu([self.model], memory_required=self.model.model.memory_required(self.shape()))
            self.samplers.cast_to_load_options(options, device=device, dtype=self.model.model_dtype())
            self.model.pre_run()
            self.result.update(dit_compute_dtype=str(self.model.model.get_dtype_inference()), patcher=type(self.model).__name__)
        self.stage("smoke.prepare_dit", prepare)
        def transfer():
            return latent.to(device)
        noise = self.stage("smoke.transfer_latent", transfer)
        count = self.config.get("spectrum_calls", 5) if self.spectrum else 1
        sigmas = torch.linspace(1.0, 0.05, count, device=device) if self.spectrum else torch.tensor([0.5], device=device)
        schedule = torch.cat((sigmas, sigmas.new_zeros(1)))
        try:
            for index, sigma in enumerate(sigmas.tolist()):
                for branch, conditioning in enumerate((positive, negative)):
                    def forward(conditioning=conditioning, branch=branch, sigma=sigma):
                        transformer_options = dict(options.get("transformer_options", {}))
                        transformer_options.update(cond_or_uncond=[branch], sample_sigmas=schedule)
                        output = self.model.model.apply_model(noise, torch.tensor([sigma], device=device),
                            c_crossattn=conditioning["cond"], transformer_options=transformer_options)
                        self.check_tensor(output, self.shape())
                    before = self.spectrum_counts()
                    self.stage(f"smoke.forward.{index}.{branch}", forward, branch=branch, sigma=sigma)
                    after = self.spectrum_counts()
                    self.result["events"][-1]["execution"] = (
                        "forecast" if after["forecast"] > before["forecast"] else "actual")
            if "compile" in self.options_enabled:
                self.stage("smoke.forward_repeat", forward, branch=1, sigma=sigma)
            counts = self.spectrum_counts()
            self.result.update(forward_calls=count*2 + int("compile" in self.options_enabled),
                               spectrum_actual=counts["actual"], spectrum_forecast=counts["forecast"])
            if self.spectrum and (counts["actual"] == 0 or counts["forecast"] == 0):
                raise ValueError(f"Spectrum 경로 미통과: {counts}")
        finally:
            def cleanup():
                self.model.cleanup()
                self.samplers.cast_to_load_options(options, device=self.model.offload_device)
            self.stage("smoke.cleanup_dit", cleanup)
        self.stage("smoke.decode_vae", lambda: self.decode(latent * 0.1))

    def prepare_sampling_model(self):
        self.sampling_model = self.model
        if "lora8" not in self.options_enabled:
            from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
            self.sampling_model = SkipLayerGuidanceDiTSimple.execute(
                self.model, 0.0, 1.0, double_layers="", single_layers="22")[0]

    def generate(self, request):
        name = request["name"]
        self.result["active_request"] = name
        self.synchronize()
        begin = time.monotonic()
        torch = self.torch
        torch.cuda.reset_peak_memory_stats()
        row = dict(request, status="RUNNING", stages_s={}, image_path=str(
            Path(self.config["output_dir"]) / f"{self.config['id']}-{name}.png"))
        self.result["scenarios"].append(row)
        self.persist()
        def measured(label, fn):
            key = name + "." + label
            try:
                return self.stage(key, fn)
            finally:
                if key in self.result["timings_s"]:
                    row["stages_s"][label] = self.result["timings_s"][key]
        try:
            positive, row["positive_cache_hit"] = measured("encode_positive", lambda: self.encode(request["prompt"]))
            negative, row["negative_cache_hit"] = measured("encode_negative", lambda: self.encode(""))
            row.update(positive_tokens=positive["cond"].shape[1], negative_tokens=negative["cond"].shape[1])
            def sample():
                pos = [[positive["cond"], {k: v for k, v in positive.items() if k != "cond"}]]
                neg = [[negative["cond"], {k: v for k, v in negative.items() if k != "cond"}]]
                return self.nodes.common_ksampler(self.sampling_model, request["seed"], request["steps"], request["cfg"],
                    "euler", "sgm_uniform", pos, neg, {"samples": torch.zeros(self.shape())}, denoise=1.0)[0]["samples"]
            samples = measured("sample", sample)
            row["dit_compute_dtype"] = str(self.model.model.get_dtype_inference())
            self.result["vae_actual_path"] = "tiled" if "tiled_vae" in self.options_enabled else "default"
            pixels = measured("decode_vae", lambda: self.decode(samples))
            def save():
                from PIL import Image
                import numpy as np
                image = Image.fromarray((pixels[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8))
                image.save(row["image_path"])
            measured("save_image", save)
            row.update(status="PASS", request_s=round(time.monotonic()-begin, 6),
                       peak_allocated_gib=round(torch.cuda.max_memory_allocated()/2**30, 3),
                       peak_reserved_gib=round(torch.cuda.max_memory_reserved()/2**30, 3))
            row["other_request_s"] = round(row["request_s"]-sum(row["stages_s"].values()), 6)
            row["vae_actual_path"] = self.result["vae_actual_path"]
            if name == "first":
                self.result.update(first_image_ready_at=time.monotonic(),
                    load_to_first_image_s=round(time.monotonic()-self.load_started, 6))
            self.persist()
        except Exception:
            row.update(status="FAIL", request_s=round(time.monotonic()-begin, 6))
            self.persist()
            raise

    def benchmark(self):
        self.stage("prepare_sampling_model", self.prepare_sampling_model)
        for request in build_requests(self.config):
            self.generate(request)

    def run(self):
        try:
            self.stage("runtime_init", self.initialize)
            with self.torch.inference_mode():
                self.load_models()
                self.track_vae_path()
                if self.config.get("mode", "smoke") == "benchmark":
                    self.benchmark()
                    self.result["status"] = "PASS_BENCHMARK"
                else:
                    self.smoke()
                    self.result["status"] = "PASS_SMOKE"
            self.result["stage"] = "done"
        except Exception as exc:
            if isinstance(exc, FloatingPointError):
                status = "NONFINITE"
            elif self.torch is not None and isinstance(exc, self.torch.cuda.OutOfMemoryError):
                status = "CUDA_OOM"
            elif isinstance(exc, MemoryError) or "Cannot allocate memory" in str(exc) or "DefaultCPUAllocator" in str(exc):
                status = "HOST_OOM"
            else:
                status = "FAIL"
            self.result.update(status=status, error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            self.result["worker_total_s"] = round(time.monotonic()-self.started, 6)
            self.persist()
            write_json(self.config["result_path"], self.result)
        return 0 if self.result["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(CandidateSession(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))).run())
