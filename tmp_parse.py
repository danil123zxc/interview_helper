import re, html
from pathlib import Path
text = Path('deepagents_backends.html').read_text(encoding='utf-8', errors='ignore')
parts = re.findall(r'"compiledSource"\s*:\s*"(.*?)"', text)
print('parts', len(parts))
if parts:
    s = bytes(parts[0], 'utf-8').decode('unicode_escape')
    s = html.unescape(s)
    print(s[:2000])
