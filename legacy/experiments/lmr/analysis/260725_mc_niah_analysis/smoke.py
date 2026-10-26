"""GPU smoke: 로드 → forward → diagnostics 확인. sbatch로 실행."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import load_mc

tok = load_mc.load_tokenizer()
ids = torch.tensor([tok("The grass is green. " * 120).input_ids[:512]], device="cuda")
for kind in ("vanilla-5B", "mc-5B", "mc-30B"):
    model = load_mc.load_model(kind)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids)
    assert logits.shape[:2] == (1, 512), logits.shape
    layers = load_mc.mc_layers(model)
    print(f"[smoke] {kind}: logits ok, mc_layers={len(layers)}")
    if layers:
        _, attn = layers[0]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.transformer.wte(ids)
            out, diag = attn.forward_with_diagnostics(model.transformer.h[0].norm_1(h))
        print(f"[smoke] diag route_indices {tuple(diag.route_indices.shape)} "
              f"n_seg={(512 + 255)//256}")
        assert diag.route_indices.shape == (1, 512, 2)
    del model; torch.cuda.empty_cache()
print("[smoke] ALL OK")
