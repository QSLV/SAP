import torch
import numpy as np
import torch.nn as nn
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from util import class2label

print(f"torch.__version__: {torch.__version__}")
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ====================================================
# Config
# ====================================================
class CFG:
    '''
    Configuration of model
    '''
    num_workers = 0
    path = "model/"
    config_path = path + 'config.pth'
    model = "roberta-base"
    batch_size = 16
    fc_dropout = 0.2
    target_size = 10
    max_len = 512
    seed = 42
    n_fold = 5
    trn_fold = [i for i in range(n_fold)]


# ====================================================
# Model
# ====================================================
class CustomModel(nn.Module):
    '''
    Customize pretrained model
    '''
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel.from_config(self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, self.cfg.target_size)
        self._init_weights(self.fc)
        self.attention = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self._init_weights(self.attention)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        # feature = torch.mean(last_hidden_states, 1)
        weights = self.attention(last_hidden_states)
        feature = torch.sum(weights * last_hidden_states, dim=1)
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output


def get_prediction(text):
    '''Transform and predict input text and return predicted class'''
    predictions = []
    inputs = CFG.tokenizer(text,
                           add_special_tokens=True,
                           max_length=CFG.max_len,
                           padding='max_length', truncation=True,
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long).reshape(1, -1)
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    for model in models_list:
        with torch.no_grad():
            y_preds = model(inputs)
        y_prob = y_preds.softmax(dim=1).to('cpu').numpy()
        predictions.append(y_prob)
        torch.cuda.empty_cache()
    predictions = np.mean(predictions, axis=0)
    predictions = np.argmax(predictions, axis=1)
    return class2label(predictions[0])


CFG.tokenizer = AutoTokenizer.from_pretrained(CFG.path + 'tokenizer')
models_list = []
for fold in CFG.trn_fold:
    print("Loading Fold {}".format(fold))
    model = CustomModel(CFG, config_path=CFG.config_path, pretrained=False)
    state = torch.load(CFG.path + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
                       map_location=torch.device('cuda'))
    model.load_state_dict(state['model'])
    model.eval()
    model.to(device)
    models_list.append(model)
    del state
