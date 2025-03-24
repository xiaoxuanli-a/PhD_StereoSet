import transformers 
import torch
from torch import nn


class LLaMALM(transformers.PreTrainedModel):
    def __init__(self, pretrained_model, checkpoint_path=None):
        pass  # Won't be used since we're overriding __new__

    def __new__(cls, pretrained_model, checkpoint_path="../../PhD_causalDPO/.cache/root/hh_dpo_llama3b_random_2025-03-23_00-13-18_121917/LATEST/policy.pt"):
        model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model)
        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}")
            raw_ckpt = torch.load(checkpoint_path, map_location="cpu")
            # Extract actual model state dict
            if "state" in raw_ckpt:
                print("state in raw_ckpt")
                state_dict = raw_ckpt["state"]
            else:
                # fallback: remove top-level keys like "metrics", "step_idx", etc.
                state_dict = {k: v for k, v in raw_ckpt.items() if k.startswith("model.")}

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        return model
    
class LLaMALM_debiased(transformers.PreTrainedModel):
    def __init__(self, pretrained_model, checkpoint_path=None):
        pass  # Won't be used since we're overriding __new__

    def __new__(cls, pretrained_model, checkpoint_path="../../PhD_causalDPO/.cache/root/hh_dpo_llama3b_random_stereo_debias_2025-03-23_06-00-07_006205/LATEST/policy.pt"):
        model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model)
        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}")
            raw_ckpt = torch.load(checkpoint_path, map_location="cpu")
            # Extract actual model state dict
            if "state" in raw_ckpt:
                print("state in raw_ckpt")
                state_dict = raw_ckpt["state"]
            else:
                # fallback: remove top-level keys like "metrics", "step_idx", etc.
                state_dict = {k: v for k, v in raw_ckpt.items() if k.startswith("model.")}

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        return model



class BertLM(transformers.BertPreTrainedModel):
    def __init__(self):
        pass

    def __new__(self, pretrained_model):
        return transformers.BertForMaskedLM.from_pretrained(pretrained_model)

class BertNextSentence(transformers.BertPreTrainedModel):
    def __init__(self, pretrained_model):
        pass

    def __new__(self, pretrained_model):
        return transformers.BertForNextSentencePrediction.from_pretrained(pretrained_model)

class RoBERTaLM(transformers.BertPreTrainedModel):
    def __init__(self, pretrained_model):
        pass

    def __new__(self, pretrained_model):
        return transformers.RobertaForMaskedLM.from_pretrained(pretrained_model)

class XLNetLM(transformers.BertPreTrainedModel):
    def __init__(self, pretrained_model):
        pass

    def __new__(self, pretrained_model):
        return transformers.XLNetLMHeadModel.from_pretrained(pretrained_model)

class XLMLM(transformers.BertPreTrainedModel):
    def __init__(self, pretrained_model):
        pass

    def __new__(self, pretrained_model):
        return transformers.XLMWithLMHeadModel.from_pretrained(pretrained_model)

class GPT2LM(transformers.GPT2PreTrainedModel):
    def __init__(self, pretrained_model):
        pass

    def __new__(self, pretrained_model):
        return transformers.GPT2LMHeadModel.from_pretrained(pretrained_model)

class ModelNSP(nn.Module):
    def __init__(self, pretrained_model, nsp_dim=300):
        super(ModelNSP, self).__init__()
        self.pretrained2model = {"xlnet": "XLNetModel", "bert": "BertModel", "roberta": "RobertaModel", "gpt2": "GPT2Model"}
        self.model_class = self.pretrained2model[pretrained_model.lower().split("-")[0]]
        self.core_model = getattr(transformers, self.model_class).from_pretrained(pretrained_model)
        self.core_model.train()
        # if pretrained_model=="gpt2-xl":
          # for name, param in self.core_model.named_parameters():
            # print(name)
            # # freeze word token embeddings and word piece embeddings!
            # if 'wte' in name or 'wpe' in name: 
              # param.requires_grad = False
        hidden_size = self.core_model.config.hidden_size
        self.nsp_head = nn.Sequential(nn.Linear(hidden_size, nsp_dim), 
            nn.Linear(nsp_dim, nsp_dim),
            nn.Linear(nsp_dim, 2))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None, \
            position_ids=None, head_mask=None, labels=None):

        if 'Roberta' in self.model_class or 'GPT2' in self.model_class:
            outputs = self.core_model(input_ids, attention_mask=attention_mask)#, token_type_ids=token_type_ids)
        else:
            outputs = self.core_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # assert len(outputs)==2

        if 'gpt2' in self.model_class.lower():
            output = outputs[0].mean(dim=1)
            logits = self.nsp_head(output)
        elif 'XLNet' in self.model_class: 
            logits = self.nsp_head(outputs[0][:,0,:]) 
        else:
            logits = self.nsp_head(outputs[1]) 

        if labels is not None:
            output = logits
            if type(output)==tuple:
                output = output[0]

            loss = self.criterion(logits, labels)
            return output, loss
        return logits 
