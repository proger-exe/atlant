def format_thinker(**kwargs):
    with open("atlant_coder/models/pt_think.md", "r+", encoding="utf-8") as prompt:
        prompt = prompt.read()
        return prompt.format(**kwargs)
