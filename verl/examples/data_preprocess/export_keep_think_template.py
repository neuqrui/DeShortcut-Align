import argparse
from pathlib import Path

from transformers import AutoTokenizer


TARGET_SNIPPET = "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model/tokenizer used for SFT.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output .jinja file path for custom chat template.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoTokenizer.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    template = tokenizer.chat_template
    if template is None:
        raise ValueError(f"No chat_template found in tokenizer from: {args.model_path}")

    replaced = template.replace(TARGET_SNIPPET, "")
    if replaced == template:
        print("[warn] target think-strip snippet not found; writing original template unchanged.")
    else:
        print("[ok] removed think-strip snippet from template.")

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(replaced, encoding="utf-8")
    print(f"[ok] wrote custom template to: {output_path}")


if __name__ == "__main__":
    main()
