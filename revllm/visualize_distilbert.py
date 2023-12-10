# todo: add docstrings
# todo: add adjust labels in box plots
# todo: replace prep_for_visualization functions - currently they're called by each individual lc_visualize_token function

import random as rn

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from bertviz import head_view, model_view
from captum.attr import LayerConductance
from transformers import (
    DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# todo: is summing the most natural attribution summary?
# todo: is normalization natural?
def summarize_attributions(attributions: torch.Tensor) -> torch.Tensor:
    """
    Summarize attributions across the sequence length dimension.
    """
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)

    return attributions


def pdf_attr(attrs, bins: int = 100) -> np.ndarray:
    """
    Compute the probability density function of the attributions.
    """
    return np.histogram(attrs, bins=bins, density=True)[0]


def select_index(tokens_list: list[str], token: str = None) -> int:
    # todo: simplify
    """
    Select the index of the token to explain.
    """
    if token is None:
        token = rn.randint(0, len(tokens_list) - 1)
        print(f"Randomly selected token: {tokens_list[token]}")

        return token
    else:
        indices = [index for index, value in enumerate(tokens_list) if value == token]

        if not indices:
            print(f"{token} not found in the list.")
            return "Not found"

        if len(indices) == 1:
            return indices[0]

        while True:
            print(f"The token you chose occurs more than once, at indices: {indices}")
            chosen_index = int(input(f"Please one of these indices: "))
            if chosen_index in indices:
                return chosen_index
            else:
                print(f"Invalid choice. Please choose an index from {indices}")


class VisualizeSentiment:
    def __init__(self, model_path: str, preprocessor):
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model_name = model_path
        self.model.to(device)
        self.model.eval()
        self.model.zero_grad()
        self.preprocessor = preprocessor
        self.layer_attrs = None
        self.layer_attributions = None
        self.layer_attrs_dist = None

    def _predict_for_visualization(
        self, input_embs: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        pred = self.model(inputs_embeds=input_embs, attention_mask=attention_mask)
        return pred.logits

    def _prep_for_visualization(self, token_to_explain_index=None):
        self.layer_attrs = []
        self.layer_attributions = None

        input_embeddings = self.model.distilbert.embeddings(self.preprocessor.input_ids)
        baseline_input_embeddings = self.model.distilbert.embeddings(
            self.preprocessor.baseline_input_ids
        )

        for i in range(self.model.config.num_hidden_layers):
            lc = LayerConductance(
                self._predict_for_visualization, self.model.distilbert.transformer.layer[i]
            )
            self.layer_attributions = lc.attribute(
                inputs=input_embeddings,
                baselines=baseline_input_embeddings,
                additional_forward_args=self.preprocessor.attention_mask,
                target=self.preprocessor.ground_truth_index,
            )
            if token_to_explain_index is not None:
                # Use only the attributions for the specified token index
                attributions_for_token = (
                    self.layer_attributions[0, token_to_explain_index].cpu().detach().tolist()
                )
                self.layer_attrs.append(attributions_for_token)
                self.layer_attrs_dist = [
                    np.array(attrs) for attrs in self.layer_attrs
                ]  # todo: this doesn't seem right, it replaces every loop?

            else:
                self.layer_attrs.append(
                    summarize_attributions(self.layer_attributions).cpu().detach().tolist()
                )
                # todo: double check on this, and make updates to above
            # if token_to_explain_index is not None:
            #     self.layer_attrs_dist = [np.array(attrs) for attrs in self.layer_attrs]

    def lc_visualize_layers(self):
        self._prep_for_visualization()
        fig, ax = plt.subplots(figsize=(15, 5))

        xticklabels = self.preprocessor.all_tokens
        yticklabels = list(range(1, len(self.layer_attrs) + 1))
        # todo: consider center=0
        ax = sns.heatmap(
            np.array(self.layer_attrs),
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            linewidth=0.2,
        )
        plt.xlabel("Tokens")
        plt.ylabel("Layers")
        plt.title("Token attribution scores for sentiment prediction")
        plt.show()

    def lc_visualize_token_boxes(self, token_to_explain: str = None):
        tokens_list = self.preprocessor.all_tokens
        token_to_explain_index = select_index(token=token_to_explain, tokens_list=tokens_list)
        if token_to_explain_index == "Not found":
            return None
        self._prep_for_visualization(token_to_explain_index=token_to_explain_index)

        fig, ax = plt.subplots(figsize=(20, 10))
        ax = sns.boxplot(data=self.layer_attrs_dist)  # Already renamed this in the previous step.
        plt.title(f"Attribution scores of {tokens_list[token_to_explain_index]} by layer")
        plt.xlabel("Layers")
        plt.ylabel("Attribution")
        plt.show()

    def lc_visualize_token_pdfs(self, token_to_explain: str = None):
        tokens_list = self.preprocessor.all_tokens
        token_to_explain_index = select_index(token=token_to_explain, tokens_list=tokens_list)
        if token_to_explain_index == "Not found":
            return None
        self._prep_for_visualization(
            token_to_explain_index=token_to_explain_index
        )  # todo: make so i don't have to call it twice

        layer_attrs_pdf = map(lambda single_attr: pdf_attr(single_attr), self.layer_attrs_dist)
        layer_attrs_pdf = np.array(list(layer_attrs_pdf))

        attr_sum = np.array(self.layer_attrs_dist).sum(-1)

        layer_attrs_pdf_norm = np.linalg.norm(layer_attrs_pdf, axis=-1, ord=1)
        layer_attrs_pdf = np.transpose(layer_attrs_pdf)
        layer_attrs_pdf = np.divide(
            layer_attrs_pdf, layer_attrs_pdf_norm, where=layer_attrs_pdf_norm != 0
        )

        # Compute PDF for the attributions.
        fig, ax = plt.subplots(figsize=(20, 10))
        plt.plot(layer_attrs_pdf)
        plt.title(f"Probability density function of {tokens_list[token_to_explain_index]}")
        plt.xlabel("Bins")
        plt.ylabel("Density")
        plt.legend(
            ["Layer " + str(i) for i in range(0, len(self.layer_attrs))]
        )  # Updated reference from layer_attrs_start
        plt.show()

    def lc_visualize_token_entropies(self, token_to_explain: str = None):
        tokens_list = self.preprocessor.all_tokens
        token_to_explain_index = select_index(token=token_to_explain, tokens_list=tokens_list)
        if token_to_explain_index == "Not found":
            return None
        self._prep_for_visualization(
            token_to_explain_index=token_to_explain_index
        )  # todo: make so i don't have to call it twice

        # todo: this is a repeat.  Better way?
        layer_attrs_pdf = map(lambda single_attr: pdf_attr(single_attr), self.layer_attrs_dist)
        layer_attrs_pdf = np.array(list(layer_attrs_pdf))

        attr_sum = np.array(self.layer_attrs_dist).sum(-1)

        layer_attrs_pdf_norm = np.linalg.norm(layer_attrs_pdf, axis=-1, ord=1)
        layer_attrs_pdf = np.transpose(layer_attrs_pdf)
        layer_attrs_pdf = np.divide(
            layer_attrs_pdf, layer_attrs_pdf_norm, where=layer_attrs_pdf_norm != 0
        )

        fig, ax = plt.subplots(figsize=(20, 10))

        layer_attrs_pdf[layer_attrs_pdf == 0] = 1
        layer_attrs_pdf_log = np.log2(layer_attrs_pdf)

        entropies = -(layer_attrs_pdf * layer_attrs_pdf_log).sum(0)

        plt.scatter(
            np.arange(len(self.layer_attrs)), attr_sum, s=entropies * 100
        )  # Updated reference from layer_attrs_start
        plt.title(f"Entropies of {tokens_list[token_to_explain_index]} by layer")
        plt.xlabel("Layers")
        plt.ylabel("Total Attribution")
        plt.show()

    def BertViz(self):
        visualizer = BertViz(model=self.model_name, preprocessor=self.preprocessor)
        visualizer.model_view_visualize()
        visualizer.head_view_visualize()
        visualizer.get_attention_weights()


class VisualizeQAndA:
    def __init__(self, model_path: str, preprocessor):
        """
        model_path: path to the model
        preprocessor: preprocessor object
        """

        self.model = DistilBertForQuestionAnswering.from_pretrained(model_path)
        self.model.to(device)
        self.model_name = model_path
        self.model.eval()
        self.model.zero_grad()
        self.preprocessor = preprocessor
        self.layer_attrs_start = []
        self.layer_attrs_end = []
        self.layer_attrs_start_dist = []
        self.layer_attrs_end_dist = []
        self.layer_attributions_start = None
        self.layer_attributions_end = None

    def _predict_for_visualization(
        self, input_embs: torch.Tensor, attention_mask: torch.Tensor, position: int = 0
    ) -> torch.Tensor:
        pred = self.model(inputs_embeds=input_embs, attention_mask=attention_mask)
        pred = pred[position]
        return pred.max(1).values

    def _prep_for_visualization(self, token_to_explain_index: int = None):
        # todo: this will not be necessary after separating out "token_to_explain_index" loop below
        self.layer_attrs_start = []
        self.layer_attrs_end = []
        self.layer_attrs_start_dist = []
        self.layer_attrs_end_dist = []
        self.layer_attributions_start = None
        self.layer_attributions_end = None

        input_embeddings = self.model.distilbert.embeddings(self.preprocessor.input_ids)
        baseline_input_embeddings = self.model.distilbert.embeddings(
            self.preprocessor.baseline_input_ids
        )

        for i in range(self.model.config.num_hidden_layers):
            lc = LayerConductance(
                self._predict_for_visualization, self.model.distilbert.transformer.layer[i]
            )
            self.layer_attributions_start = lc.attribute(
                inputs=input_embeddings,
                baselines=baseline_input_embeddings,
                additional_forward_args=(self.preprocessor.attention_mask, 0),
            )
            self.layer_attributions_end = lc.attribute(
                inputs=input_embeddings,
                baselines=baseline_input_embeddings,
                additional_forward_args=(self.preprocessor.attention_mask, 1),
            )
            self.layer_attrs_start.append(
                summarize_attributions(self.layer_attributions_start).cpu().detach().tolist()
            )
            self.layer_attrs_end.append(
                summarize_attributions(self.layer_attributions_end).cpu().detach().tolist()
            )

            # todo: separate this out so I don't need to run _prep_for_visualization multiple times below
            if token_to_explain_index is not None:
                self.layer_attrs_start_dist.append(
                    self.layer_attributions_start[0, token_to_explain_index, :]
                    .cpu()
                    .detach()
                    .tolist()
                )
                self.layer_attrs_end_dist.append(
                    self.layer_attributions_end[0, token_to_explain_index, :]
                    .cpu()
                    .detach()
                    .tolist()
                )

    def lc_visualize_layers(self):
        # todo: subdivide

        self._prep_for_visualization()

        fig, ax = plt.subplots(figsize=(15, 5))
        xticklabels = self.preprocessor.all_tokens
        yticklabels = list(range(1, len(self.layer_attrs_start) + 1))
        ax = sns.heatmap(
            np.array(self.layer_attrs_start),
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            linewidth=0.2,
        )
        plt.xlabel("Tokens")
        plt.ylabel("Layers")
        plt.title("Token attribution scores by layer for start of answer")
        plt.show()

        fig, ax = plt.subplots(figsize=(15, 5))
        xticklabels = self.preprocessor.all_tokens
        yticklabels = list(range(1, len(self.layer_attrs_start) + 1))
        ax = sns.heatmap(
            np.array(self.layer_attrs_end),
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            linewidth=0.2,
        )  # , annot=True
        plt.xlabel("Tokens")
        plt.ylabel("Layers")
        plt.title("Token attribution scores by layer for end of answer")
        plt.show()

    def lc_visualize_token_boxes(self, token_to_explain: str = None):
        # todo: add option to visualize multiple tokens at once
        # todo: take str as input for token to explain
        # todo: subdivide
        tokens_list = self.preprocessor.all_tokens
        token_to_explain_index = select_index(token=token_to_explain, tokens_list=tokens_list)
        if token_to_explain_index == "Not found":
            return None
        self._prep_for_visualization(
            token_to_explain_index=token_to_explain_index
        )  # todo: make so i don't have to call it twice

        fig, ax = plt.subplots(figsize=(20, 10))
        ax = sns.boxplot(data=self.layer_attrs_start_dist)
        plt.title(
            f"Attribution scores of {tokens_list[token_to_explain_index]} for start of answer"
        )
        plt.xlabel("Layers")
        plt.ylabel("Attribution")
        plt.show()

        fig, ax = plt.subplots(figsize=(20, 10))
        ax = sns.boxplot(data=self.layer_attrs_end_dist)
        plt.title(f"Attribution scores of {tokens_list[token_to_explain_index]} for end of answer")
        plt.xlabel("Layers")
        plt.ylabel("Attribution")
        plt.show()

    def lc_visualize_token_pdfs(self, token_to_explain: str = None):
        tokens_list = self.preprocessor.all_tokens
        token_to_explain_index = select_index(token=token_to_explain, tokens_list=tokens_list)
        if token_to_explain_index == "Not found":
            return None
        self._prep_for_visualization(
            token_to_explain_index=token_to_explain_index
        )  # todo: make so i don't have to call it twice

        layer_attrs_end_pdf = map(
            lambda single_attr: pdf_attr(single_attr), self.layer_attrs_end_dist
        )
        layer_attrs_end_pdf = np.array(list(layer_attrs_end_pdf))

        attr_sum = np.array(self.layer_attrs_end_dist).sum(-1)

        # size: #layers
        layer_attrs_end_pdf_norm = np.linalg.norm(layer_attrs_end_pdf, axis=-1, ord=1)

        # size: #bins x #layers
        layer_attrs_end_pdf = np.transpose(layer_attrs_end_pdf)

        # size: #bins x #layers
        layer_attrs_end_pdf = np.divide(
            layer_attrs_end_pdf, layer_attrs_end_pdf_norm, where=layer_attrs_end_pdf_norm != 0
        )

        fig, ax = plt.subplots(figsize=(20, 10))
        plt.plot(layer_attrs_end_pdf)
        plt.title(
            f"Probability density function of {tokens_list[token_to_explain_index]} for end position"
        )
        plt.xlabel("Bins")
        plt.ylabel("Density")
        plt.legend(["Layer " + str(i) for i in range(0, len(self.layer_attrs_start))])
        plt.show()

    def lc_visualize_token_entropies(self, token_to_explain: str = None):
        tokens_list = self.preprocessor.all_tokens
        token_to_explain_index = select_index(token=token_to_explain, tokens_list=tokens_list)
        if token_to_explain_index == "Not found":
            return None
        self._prep_for_visualization(
            token_to_explain_index=token_to_explain_index
        )  # todo: make so i don't have to call it twice

        layer_attrs_end_pdf = map(
            lambda single_attr: pdf_attr(single_attr), self.layer_attrs_end_dist
        )
        layer_attrs_end_pdf = np.array(list(layer_attrs_end_pdf))

        attr_sum = np.array(self.layer_attrs_end_dist).sum(-1)

        # size: #layers
        layer_attrs_end_pdf_norm = np.linalg.norm(layer_attrs_end_pdf, axis=-1, ord=1)

        # size: #bins x #layers
        layer_attrs_end_pdf = np.transpose(layer_attrs_end_pdf)

        # size: #bins x #layers
        layer_attrs_end_pdf = np.divide(
            layer_attrs_end_pdf, layer_attrs_end_pdf_norm, where=layer_attrs_end_pdf_norm != 0
        )

        fig, ax = plt.subplots(figsize=(20, 10))

        # replacing 0s with 1s. np.log(1) = 0 and np.log(0) = -inf
        layer_attrs_end_pdf[layer_attrs_end_pdf == 0] = 1
        layer_attrs_end_pdf_log = np.log2(layer_attrs_end_pdf)

        # size: #layers
        entropies = -(layer_attrs_end_pdf * layer_attrs_end_pdf_log).sum(0)

        plt.scatter(np.arange(len(self.layer_attrs_start)), attr_sum, s=entropies * 100)
        plt.title(f"Entropies of {tokens_list[token_to_explain_index]} by layer for end position")
        plt.xlabel("Layers")
        plt.ylabel("Total Attribution")
        plt.show()

    def BertViz(self):
        visualizer = BertViz(model=self.model_name, preprocessor=self.preprocessor)
        visualizer.model_view_visualize()
        visualizer.head_view_visualize()
        visualizer.get_attention_weights()


# todo: does not currently work.
class VisualizeMaskedLM:
    def __init__(self, model_path: str, preprocessor):
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model_name = model_path
        self.model.eval()
        self.model.zero_grad()
        self.preprocessor = preprocessor
        self.layer_attrs = []
        self.layer_attributions = None

    def BertViz(self):
        visualizer = BertViz(model=self.model_name, preprocessor=self.preprocessor)
        visualizer.model_view_visualize()
        visualizer.head_view_visualize()
        visualizer.get_attention_weights()

    def _predict_for_visualization(
        self, input_embs: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        pred = self.model(inputs_embeds=input_embs, attention_mask=attention_mask)
        return pred.logits

    def _prep_for_visualization(self, token_to_explain_index=None):
        self.layer_attrs = []
        self.layer_attributions = None

        input_embeddings = self.model.distilbert.embeddings(self.preprocessor.input_ids)
        baseline_input_embeddings = self.model.distilbert.embeddings(
            self.preprocessor.baseline_input_ids
        )

        for i in range(self.model.config.num_hidden_layers):
            lc = LayerConductance(
                self._predict_for_visualization, self.model.distilbert.transformer.layer[i]
            )
            self.layer_attributions = lc.attribute(
                inputs=input_embeddings,
                baselines=baseline_input_embeddings,
                additional_forward_args=self.preprocessor.attention_mask,
                target=(self.preprocessor.mask_index, self.preprocessor.ground_truth_index),
            )
            if token_to_explain_index is not None:
                # Use only the attributions for the specified token index
                attributions_for_token = (
                    self.layer_attributions[0, token_to_explain_index].cpu().detach().tolist()
                )
                self.layer_attrs.append(attributions_for_token)
            else:
                self.layer_attrs.append(
                    summarize_attributions(self.layer_attributions).cpu().detach().tolist()
                )

                # todo: separate this out so I don't need to run _prep_for_visualization multiple times below
            if token_to_explain_index is not None:
                self.layer_attrs_dist = [np.array(attrs) for attrs in self.layer_attrs]

    def lc_visualize_layers(self):
        self._prep_for_visualization()
        fig, ax = plt.subplots(figsize=(15, 5))
        xticklabels = self.preprocessor.all_tokens
        yticklabels = list(range(1, len(self.layer_attrs) + 1))
        ax = sns.heatmap(
            np.array(self.layer_attrs),
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            linewidth=0.2,
        )
        plt.xlabel("Tokens")
        plt.ylabel("Layers")
        plt.title("Token attribution scores for sentiment prediction")
        plt.show()

    def lc_visualize_token(self, token_to_explain: str = None):
        tokens_list = self.preprocessor.all_tokens
        token_to_explain_index = select_index(token=token_to_explain, tokens_list=tokens_list)
        if token_to_explain_index == "Not found":
            return None
        self._prep_for_visualization(
            token_to_explain_index=token_to_explain_index
        )  # todo: make so i don't have to call it twice

        fig, ax = plt.subplots(figsize=(20, 10))
        ax = sns.boxplot(data=self.layer_attrs_dist)  # Already renamed this in the previous step.
        plt.title(
            f"Attribution scores of {tokens_list[token_to_explain_index]} for sentiment prediction"
        )
        plt.xlabel("Layers")
        plt.ylabel("Attribution")
        plt.show()

        # Compute PDF for the attributions.
        layer_attrs_pdf = map(lambda single_attr: pdf_attr(single_attr), self.layer_attrs_dist)
        layer_attrs_pdf = np.array(list(layer_attrs_pdf))

        attr_sum = np.array(self.layer_attrs_dist).sum(-1)

        layer_attrs_pdf_norm = np.linalg.norm(layer_attrs_pdf, axis=-1, ord=1)
        layer_attrs_pdf = np.transpose(layer_attrs_pdf)
        layer_attrs_pdf = np.divide(
            layer_attrs_pdf, layer_attrs_pdf_norm, where=layer_attrs_pdf_norm != 0
        )

        fig, ax = plt.subplots(figsize=(20, 10))
        plt.plot(layer_attrs_pdf)
        plt.xlabel("Bins")
        plt.ylabel("Density")
        plt.legend(
            ["Layer " + str(i) for i in range(0, len(self.layer_attrs))]
        )  # Updated reference from layer_attrs_start
        plt.show()

        fig, ax = plt.subplots(figsize=(20, 10))

        layer_attrs_pdf[layer_attrs_pdf == 0] = 1
        layer_attrs_pdf_log = np.log2(layer_attrs_pdf)

        entropies = -(layer_attrs_pdf * layer_attrs_pdf_log).sum(0)

        plt.scatter(
            np.arange(len(self.layer_attrs)), attr_sum, s=entropies * 100
        )  # Updated reference from layer_attrs_start
        plt.xlabel("Layers")
        plt.ylabel("Total Attribution")
        plt.show()


class BertViz:
    def __init__(self, model: str, preprocessor):
        self.preprocessor = preprocessor
        self.attention_mask = self.preprocessor.attention_mask
        self.tokenizer = self.preprocessor.tokenizer
        self.input_ids = self.preprocessor.input_ids
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model, output_attentions=True
        )
        self.attention_weights = None
        self.tokens = None

    def _run_model(self):
        outputs = self.model(self.input_ids, attention_mask=self.attention_mask)
        attention = outputs.attentions

        self.attention_weights = attention
        self.tokens = self.tokenizer.convert_ids_to_tokens(self.input_ids[0].tolist())

    def model_view_visualize(self):
        if self.attention_weights is None or self.tokens is None:
            self._run_model()
        model_view(self.attention_weights, self.tokens)

    def head_view_visualize(self):
        if self.attention_weights is None or self.tokens is None:
            self._run_model()
        head_view(self.attention_weights, self.tokens)

    def get_attention_weights(self):
        if self.attention_weights is None:
            self._run_model()
        return self.attention_weights
