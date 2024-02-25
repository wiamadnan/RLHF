from dataclasses import dataclass, field
from typing import Optional

def print_config(config, config_name):
    print(f"--- {config_name} ---")
    for attr in dir(config):
        # Filter out private attributes and methods
        if not attr.startswith("_") and not callable(getattr(config, attr)):
            print(f"{attr}: {getattr(config, attr)}")
    print("\n")


@dataclass
class ScriptArguments:
    use_seq2seq: bool = field(default=False, metadata={"help": "whether to use seq2seq"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})

    # LoraConfig
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})
