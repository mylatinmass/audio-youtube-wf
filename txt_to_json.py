import re
import json
from datetime import timedelta

def txt_to_json(txt_path, json_path):
    """
    Read lines like “[HH:MM:SS - HH:MM:SS] text...” and build:
      {
        "text": full transcript,
        "segments": [
          {"start": sec, "end": sec, "text": "..."},
          …
        ]
      }
    """
    pattern = re.compile(
        r"\[(\d+):(\d+):(\d+)\s*-\s*(\d+):(\d+):(\d+)\]\s*(.*)"
    )
    segments = []
    full_text_parts = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            m = pattern.match(line)
            if not m:
                continue
            h1, m1, s1, h2, m2, s2, txt = m.groups()
            start = int(h1)*3600 + int(m1)*60 + int(s1)
            end   = int(h2)*3600 + int(m2)*60 + int(s2)
            segments.append({
                "id": idx,
                "start": start,
                "end":   end,
                "text":  txt,
            })
            full_text_parts.append(txt)

    result = {
        "text": "\n".join(full_text_parts),
        "segments": segments
    }
    # write it out
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result
