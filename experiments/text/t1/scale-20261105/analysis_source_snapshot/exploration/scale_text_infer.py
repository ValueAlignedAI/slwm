"""Raw continuation from a saved early scale checkpoint; no training corpus needed."""
import argparse
import json
import torch
import tiktoken
from training.scale_text_train import initialize


def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--prompt",required=True); p.add_argument("--tokens",type=int,default=64); p.add_argument("--seed",type=int,default=81); p.add_argument("--device",choices=("cpu","cuda"),default="cpu"); p.add_argument("--output"); a=p.parse_args(); torch.set_num_threads(4)
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False); model=initialize(ck["config"],ck["run"]["variant"],ck["run"]["seed"]); model.load_state_dict(ck["model"],strict=True); model.to(a.device).eval(); enc=tiktoken.get_encoding("gpt2"); ids=torch.tensor([enc.encode(a.prompt)],device=a.device); torch.manual_seed(a.seed)
    if not ids.numel(): p.error("Prompt must contain at least one token")
    with torch.inference_mode():
        for _ in range(a.tokens):
            logits=model(ids[:,-512:])[:,-1].float()/.8; threshold=logits.topk(50).values[:,-1:]; logits.masked_fill_(logits<threshold,-float("inf")); next_id=torch.multinomial(logits.softmax(-1),1); ids=torch.cat((ids,next_id),1)
            if int(next_id)==enc.eot_token: break
    record={"run":ck["run"],"selected_step":ck["step"],"prompt":a.prompt,"text":enc.decode(ids[0].tolist()),"scope":"Raw early-model continuation;16.8M training tokens, no factual/reasoning or converged-quality claim"}
    if a.output:
        from pathlib import Path
        Path(a.output).write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record))


if __name__=="__main__": main()
