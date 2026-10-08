"""Create tiny local fixtures, with no downloads."""
import json
from pathlib import Path


def write_data(cfg):
    root=Path(cfg["datasets"]["stories"]["path"]).parent
    root.mkdir(parents=True,exist_ok=True)
    (root/"stories.txt").write_text("A happy cat found a friend. The little dog went home.\n"*200,encoding="utf-8")
    rows=[dict(prompt="Tell a story about a cat",response="The cat found a dog. They went home and played together.") for _ in range(50)]
    (root/"instruct.jsonl").write_text("\n".join(json.dumps(r) for r in rows),encoding="utf-8")


def main():
    from lithelm.config import load_config, save_config
    from lithelm.workspace import run_directory
    from lithelm.tokenizer import prepare_tokenizer
    from lithelm.data import prepare_data
    cfg = load_config(Path(__file__).resolve().parents[2] / "configs/base.yaml")
    with run_directory(cfg) as (cfg, root):
        write_data(cfg)
        tokenizer = prepare_tokenizer(cfg)
        prepare_data(cfg, tokenizer)
        from lithelm.provenance import record_environment
        record_environment(cfg, tokenizer)
        save_config(cfg, "resolved_config.yaml")
        print(root)


if __name__ == "__main__":
    main()
