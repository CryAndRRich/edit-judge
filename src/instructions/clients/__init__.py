"""Open VLM clients for the request analysis. Register each as name -> "module:Class" (module relative to this
package); modules are imported only when a client is used, so torch loads only then. To add a model, subclass
clients.base.Client in its own file and register it here."""

import importlib

CLIENTS: dict[str, str] = {
    "qwen2.5-vl-3b": ".qwen_vl:Qwen25VL3B",
    "qwen2.5-vl-7b": ".qwen_vl:Qwen25VL7B",
}


def get_client(name: str):
    target = CLIENTS.get(name)
    if target is None:
        raise SystemExit(f"no client {name!r}; registered: {sorted(CLIENTS)}")
    module, cls = target.split(":")
    return getattr(importlib.import_module(module, __name__), cls)()
