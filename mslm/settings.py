"""Typed read access to the resolved config (env variables exported by the scripts)."""

from __future__ import annotations

import os

from mslm import envfile


class Settings:
    def __init__(self, values: dict[str, str]):
        self.values = dict(values)

    @classmethod
    def from_env(cls) -> "Settings":
        """Values exported by scripts/lib/config.sh (falls back to resolving CONFIG)."""
        keys = envfile.declared_keys()
        if all(k in os.environ for k in keys):
            return cls({k: os.environ[k] for k in keys})
        return cls(envfile.resolve(os.environ.get("CONFIG") or None))

    @classmethod
    def from_config(cls, config: str | None, overrides: dict[str, str] | None = None) -> "Settings":
        env = dict(os.environ)
        for k in envfile.declared_keys():   # ignore stray shell values unless given explicitly
            env.pop(k, None)
        env.update(overrides or {})
        return cls(envfile.resolve(config, env))

    def str(self, key: str) -> str:
        return self.values[key]

    def int(self, key: str) -> int:
        return int(float(self.values[key]))   # accepts 1e9-style

    def float(self, key: str) -> float:
        return float(self.values[key])

    def bool(self, key: str) -> bool:
        return self.values[key].strip().lower() in ("1", "true", "yes", "on")

    def list(self, key: str) -> list[str]:
        return [x.strip() for x in self.values[key].split(",") if x.strip()]

    def run_name(self) -> str:
        tag = self.values.get("RUN_TAG", "")
        return self.values["RUN_NAME"] or (
            f"{self.values['MODEL_ARCH']}{'_' + tag if tag else ''}_{self.values['MODEL_SIZE']}"
            f"_ctx{self.int('SEQ_LEN')}_s{self.int('SEED')}")

    def run_dir(self) -> str:
        return os.path.join(self.values["RUNS_ROOT"], self.run_name())
