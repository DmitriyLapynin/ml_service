from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PostprocessNode:
    telemetry: Any | None = None

    # default policies
    default_voice_max_chars: int = 280
    default_email_signature: str = "—\nС уважением,\nКоманда"
    default_email_greeting: str = "Здравствуйте!"
    default_email_subject: str = "Сообщение"

    async def __call__(self, state) -> Any:
        runtime: Dict[str, Any] = getattr(state, "runtime", {}) or {}
        runtime.setdefault("executed", [])
        runtime.setdefault("errors", [])
        runtime.setdefault("degraded", False)

        runtime["executed"].append("postprocess")

        channel = getattr(state, "channel", "chat")
        route = getattr(state, "route", None) or channel
        policies: Dict[str, Any] = getattr(state, "policies", {}) or {}

        # достаём текущий ответ
        answer_obj = getattr(state, "answer", None)
        text = None
        fmt = None
        meta: Dict[str, Any] = {}

        try:
            if isinstance(answer_obj, dict):
                text = (answer_obj.get("text") or "").strip()
                fmt = answer_obj.get("format") or None
                meta = answer_obj.get("meta") or {}
            else:
                # если у тебя answer — объект, пробуем атрибуты
                text = getattr(answer_obj, "text", None)
                fmt = getattr(answer_obj, "format", None)
                meta = getattr(answer_obj, "meta", None) or {}
                text = (text or "").strip()

            if not text:
                # нечего постпроцессить
                setattr(state, "runtime", runtime)
                return state

            if route == "email":
                out = self._postprocess_email(text=text, policies=policies)
                fmt = "email"
                text = out["text"]
                meta.update(out.get("meta", {}))

            elif route == "voice":
                out = self._postprocess_voice(text=text, policies=policies)
                fmt = out.get("format", "voice")
                text = out["text"]
                meta.update(out.get("meta", {}))

            else:
                # chat / default
                out = self._postprocess_chat(text=text, policies=policies)
                fmt = fmt or "plain"
                text = out["text"]
                meta.update(out.get("meta", {}))

            # сохраняем обратно
            new_answer = {"text": text, "format": fmt, "meta": meta}
            setattr(state, "answer", new_answer)

            runtime["postprocess"] = {
                "route": route,
                "format": fmt,
                "len_chars": len(text),
            }

            if self.telemetry:
                try:
                    self.telemetry.log_step(
                        trace_id=getattr(state, "trace_id", None),
                        node="postprocess",
                        meta={"route": route, "format": fmt, "len_chars": len(text)},
                    )
                except Exception:
                    pass

            setattr(state, "runtime", runtime)
            return state

        except Exception as e:
            # никогда не падаем из-за постпроцесса
            runtime["degraded"] = True
            runtime["errors"].append({"node": "postprocess", "type": "postprocess_error", "msg": str(e)})
            setattr(state, "runtime", runtime)
            return state

    # ----------------------------
    # Chat
    # ----------------------------
    def _postprocess_chat(self, *, text: str, policies: Dict[str, Any]) -> Dict[str, Any]:
        # легкая чистка
        cleaned = text.strip()

        # опционально — ограничение длины
        max_chars = policies.get("chat_max_chars")
        if isinstance(max_chars, int) and max_chars > 0 and len(cleaned) > max_chars:
            cleaned = cleaned[: max_chars].rstrip() + "…"

        return {"text": cleaned, "meta": {"postprocess": "chat"}}

    # ----------------------------
    # Email
    # ----------------------------
    def _postprocess_email(self, *, text: str, policies: Dict[str, Any]) -> Dict[str, Any]:
        greeting = policies.get("email_greeting") or self.default_email_greeting
        signature = policies.get("email_signature") or self.default_email_signature
        subject = policies.get("email_subject") or self.default_email_subject

        # если модель уже написала приветствие — не добавляем
        has_greeting = bool(re.match(r"^(здравств(уй|уйте)|добрый\s(день|вечер)|привет)\b", text.strip(), re.I))
        body = text.strip()

        if not has_greeting:
            body = f"{greeting}\n\n{body}"

        # если подписи нет — добавим
        has_signature = "с уважением" in body.lower() or body.strip().endswith(signature.strip())
        if not has_signature:
            body = f"{body}\n\n{signature}"

        # subject кладём в meta (а не в text), чтобы backend мог вставить как надо
        return {
            "text": body,
            "meta": {"postprocess": "email", "subject": subject},
        }

    # ----------------------------
    # Voice
    # ----------------------------
    def _postprocess_voice(self, *, text: str, policies: Dict[str, Any]) -> Dict[str, Any]:
        max_chars = policies.get("voice_max_chars")
        if not isinstance(max_chars, int) or max_chars <= 0:
            max_chars = self.default_voice_max_chars

        # чистим лишние символы для речи
        cleaned = text.strip()
        cleaned = re.sub(r"\s+", " ", cleaned)

        # убираем маркдаун/списки, которые плохо читаются голосом
        cleaned = re.sub(r"[*_`#>-]+", "", cleaned).strip()

        if len(cleaned) > max_chars:
            cleaned = cleaned[: max_chars].rstrip()
            # стараемся закончить на конце предложения
            cleaned = re.sub(r"[,:;]\s*$", "", cleaned)
            cleaned += "."

        # опциональный SSML
        ssml_enabled = bool(policies.get("voice_ssml", False))
        if ssml_enabled:
            ssml = self._to_ssml(cleaned)
            return {"text": ssml, "format": "ssml", "meta": {"postprocess": "voice", "ssml": True}}

        return {"text": cleaned, "format": "voice", "meta": {"postprocess": "voice", "ssml": False}}

    def _to_ssml(self, text: str) -> str:
        # минимальный SSML (без сложных пауз)
        safe = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        # лёгкие паузы после точек
        safe = safe.replace(". ", ".<break time='200ms'/> ")
        return f"<speak>{safe}</speak>"