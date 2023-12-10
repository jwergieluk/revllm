# todo: for all token visualizations, break if entry not found in list of tokens

import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
from transformers import (
    DistilBertForMaskedLM,
    DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification,
)


def summarize_attributions(attributions: torch.Tensor) -> torch.Tensor:
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)

    return attributions


def get_top_k_attributed_tokens(
    attrs: torch.Tensor, k: int, preprocessor
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    values, indices = torch.topk(attrs, k)
    top_tokens = [preprocessor.all_tokens[idx] for idx in indices]

    return top_tokens, values, indices


class AnalyzeSentiment:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, model_name: str, preprocessor: object):
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.model.zero_grad()
        self.logits = None
        self.preprocessor = preprocessor

    def predict(self, input_ids=None, attention_mask=None) -> list:
        if input_ids is None:
            output = self.model(
                self.preprocessor.input_ids, attention_mask=self.preprocessor.attention_mask
            )

            self.logits = output.logits
            probs = torch.nn.functional.softmax(self.logits, dim=-1)
            predicted_classes = torch.argmax(probs, dim=-1).tolist()
            predictions = [self.preprocessor.labels[p] for p in predicted_classes]

            print("        Context: ", self.preprocessor.context)
            print("Predicted Answer: ", predictions[0])
            print("   Actual Answer: ", self.preprocessor.ground_truth)

        # for use in lig.attribute below, which need these args to be passed in
        else:
            output = self.model(input_ids, attention_mask=attention_mask)

            self.logits = output.logits

            return self.logits

    def lig_color_map(self):
        # Initialize LayerIntegratedGradients
        lig = LayerIntegratedGradients(self.predict, self.model.distilbert.embeddings)

        # Get attributions for the sentiment prediction
        attributions, delta = lig.attribute(
            inputs=self.preprocessor.input_ids,
            baselines=self.preprocessor.baseline_input_ids,
            additional_forward_args=self.preprocessor.attention_mask,
            return_convergence_delta=True,
            target=self.preprocessor.ground_truth_index,
        )

        attributions_sum = summarize_attributions(attributions)

        # Use the stored logits
        sentiment_score = torch.softmax(self.logits[0], dim=0)
        predicted_class = torch.argmax(sentiment_score)

        sentiment_vis = viz.VisualizationDataRecord(
            attributions_sum,
            torch.max(sentiment_score),
            predicted_class,
            predicted_class,
            str(self.preprocessor.ground_truth_index),
            attributions_sum.sum(),
            self.preprocessor.all_tokens,
            delta,
        )

        print("\033[1m", "Visualizations For Sentiment Prediction", "\033[0m")
        viz.visualize_text([sentiment_vis])

    def lig_top_k_tokens(self, k: int = 5) -> None:
        lig = LayerIntegratedGradients(
            self.predict, [self.model.distilbert.embeddings.word_embeddings]
        )

        attributions = lig.attribute(
            inputs=self.preprocessor.input_ids,
            baselines=self.preprocessor.baseline_input_ids,
            additional_forward_args=self.preprocessor.attention_mask,
            target=self.preprocessor.ground_truth_index,
        )

        attributions_word = summarize_attributions(attributions[0])
        top_words, top_words_val, top_word_ind = get_top_k_attributed_tokens(
            attributions_word, k=k, preprocessor=self.preprocessor
        )

        df = pd.DataFrame(
            {
                "Word(Index), Attribution": [
                    "{} ({}), {}".format(word, pos, round(val.item(), 2))
                    for word, pos, val in zip(top_words, top_word_ind, top_words_val)
                ]
            }
        )
        df.style.set_properties(cell_ids=False)

        full_token_list = [
            "{}({})".format(token, str(i)) for i, token in enumerate(self.preprocessor.all_tokens)
        ]

        print(f"Full token list: {full_token_list}")
        print(f"Top {k} attributed embeddings for sentiment prediction: {df}")


class AnalyzeQAndA:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, model_name: str, preprocessor: object):
        self.model = DistilBertForQuestionAnswering.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.model.zero_grad()
        self.preprocessor = preprocessor
        self.start_scores = None
        self.end_scores = None

    def predict(self, input_ids=None, attention_mask=None, position: int = 0) -> None:
        if input_ids is None:
            output = self.model(
                self.preprocessor.input_ids, attention_mask=self.preprocessor.attention_mask
            )

            self.start_scores = output.start_logits
            self.end_scores = output.end_logits

            print("        Question: ", self.preprocessor.question)
            print(
                "Predicted Answer: ",
                " ".join(
                    self.preprocessor.all_tokens[
                        torch.argmax(self.start_scores) : torch.argmax(self.end_scores) + 1
                    ]
                ),
            )
            print("   Actual Answer: ", self.preprocessor.ground_truth)

        # for use in lig.attribute below, which need these args to be passed in
        else:
            output = self.model(input_ids, attention_mask=attention_mask)

            self.start_scores = output.start_logits
            self.end_scores = output.end_logits

            pred = (self.start_scores, self.end_scores)
            pred = pred[position]
            return pred.max(1).values

    def lig_color_map(self):
        lig = LayerIntegratedGradients(self.predict, self.model.distilbert.embeddings)

        attributions_start, delta_start = lig.attribute(
            inputs=self.preprocessor.input_ids,
            baselines=self.preprocessor.baseline_input_ids,
            additional_forward_args=(self.preprocessor.attention_mask, 0),
            return_convergence_delta=True,
        )
        attributions_end, delta_end = lig.attribute(
            inputs=self.preprocessor.input_ids,
            baselines=self.preprocessor.baseline_input_ids,
            additional_forward_args=(self.preprocessor.attention_mask, 1),
            return_convergence_delta=True,
        )

        attributions_start_sum = summarize_attributions(attributions_start)
        attributions_end_sum = summarize_attributions(attributions_end)

        # storing couple samples in an array for visualization purposes
        start_position_vis = viz.VisualizationDataRecord(
            attributions_start_sum,
            torch.max(torch.softmax(self.start_scores[0], dim=0)),
            torch.argmax(self.start_scores),
            torch.argmax(self.start_scores),
            str(self.preprocessor.ground_truth_start_index),
            attributions_start_sum.sum(),
            self.preprocessor.all_tokens,
            delta_start,
        )

        end_position_vis = viz.VisualizationDataRecord(
            attributions_end_sum,
            torch.max(torch.softmax(self.end_scores[0], dim=0)),
            torch.argmax(self.end_scores),
            torch.argmax(self.end_scores),
            str(self.preprocessor.ground_truth_end_index),
            attributions_end_sum.sum(),
            self.preprocessor.all_tokens,
            delta_end,
        )

        print("\033[1m", "Visualizations For Start Position", "\033[0m")
        viz.visualize_text([start_position_vis])

        print("\033[1m", "Visualizations For End Position", "\033[0m")
        viz.visualize_text([end_position_vis])

    def lig_top_k_tokens(self, k: int = 5) -> None:
        lig = LayerIntegratedGradients(
            self.predict, [self.model.distilbert.embeddings.word_embeddings]
        )

        attributions_start = lig.attribute(
            inputs=self.preprocessor.input_ids,
            baselines=self.preprocessor.baseline_input_ids,
            additional_forward_args=(self.preprocessor.attention_mask, 0),
        )
        attributions_end = lig.attribute(
            inputs=self.preprocessor.input_ids,
            baselines=self.preprocessor.baseline_input_ids,
            additional_forward_args=(self.preprocessor.attention_mask, 1),
        )

        attributions_start_word = summarize_attributions(attributions_start[0])
        attributions_end_word = summarize_attributions(attributions_end[0])

        top_words_start, top_words_val_start, top_word_ind_start = get_top_k_attributed_tokens(
            attributions_start_word, k=k, preprocessor=self.preprocessor
        )
        top_words_end, top_words_val_end, top_words_ind_end = get_top_k_attributed_tokens(
            attributions_end_word, k=k, preprocessor=self.preprocessor
        )

        df_start = pd.DataFrame(
            {
                "Word(Index), Attribution": [
                    "{} ({}), {}".format(word, pos, round(val.item(), 2))
                    for word, pos, val in zip(
                        top_words_start, top_word_ind_start, top_words_val_start
                    )
                ]
            }
        )
        df_start.style.set_properties(cell_ids=False)

        df_end = pd.DataFrame(
            {
                "Word(Index), Attribution": [
                    "{} ({}), {}".format(word, pos, round(val.item(), 2))
                    for word, pos, val in zip(top_words_end, top_words_ind_end, top_words_val_end)
                ]
            }
        )
        df_end.style.set_properties(cell_ids=False)

        full_token_list = [
            "{}({})".format(token, str(i)) for i, token in enumerate(self.preprocessor.all_tokens)
        ]

        print(f"Full token list: {full_token_list}")
        print(f"Top 5 attributed embeddings for start position: {df_start}")
        print(f"Top 5 attributed embeddings for end position: {df_end}")


class AnalyzeMaskedLM:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, model_name: str, preprocessor: object):
        self.model = DistilBertForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.model.zero_grad()
        self.preprocessor = preprocessor
        self.logits = None

    def predict(self, input_ids=None, attention_mask=None) -> list:
        if input_ids is None:
            output = self.model(self.preprocessor.input_ids, self.preprocessor.attention_mask)
            self.logits = output.logits
            probs = torch.nn.functional.softmax(self.logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1).tolist()
            mask_prediction_id = predictions[0][self.preprocessor.mask_index]
            mask_prediction = self.preprocessor.tokenizer.convert_ids_to_tokens(mask_prediction_id)

            print("         Context: ", self.preprocessor.masked_context)
            print("    Ground truth: ", self.preprocessor.ground_truth)
            print("Predicted answer: ", mask_prediction)

        # for use in lig.attribute below, which need these args to be passed in
        else:
            output = self.model(input_ids, attention_mask=attention_mask)

            self.logits = output.logits

            return self.logits

    def lig_color_map(self):
        # Initialize LayerIntegratedGradients
        lig = LayerIntegratedGradients(self.predict, self.model.distilbert.embeddings)

        # Get attributions for the prediction
        attributions, delta = lig.attribute(
            inputs=self.preprocessor.input_ids,
            baselines=self.preprocessor.baseline_input_ids,
            additional_forward_args=self.preprocessor.attention_mask,
            return_convergence_delta=True,
            target=(self.preprocessor.mask_index, self.preprocessor.ground_truth_index),
        )

        attributions_sum = summarize_attributions(attributions)

        # Use the stored logits
        probs = torch.nn.functional.softmax(self.logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1).tolist()
        mask_prediction_id = predictions[0][self.preprocessor.mask_index]
        mask_prediction = self.preprocessor.tokenizer.convert_ids_to_tokens(mask_prediction_id)

        sentiment_vis = viz.VisualizationDataRecord(
            attributions_sum,
            torch.max(probs),
            mask_prediction,
            mask_prediction,
            str(self.preprocessor.ground_truth),
            attributions_sum.sum(),
            self.preprocessor.all_tokens,
            delta,
        )

        print("\033[1m", "Visualizations For Sentiment Prediction", "\033[0m")
        viz.visualize_text([sentiment_vis])

    def lig_top_k_tokens(self, k: int = 5) -> None:
        lig = LayerIntegratedGradients(
            self.predict, [self.model.distilbert.embeddings.word_embeddings]
        )

        attributions = lig.attribute(
            inputs=self.preprocessor.input_ids,
            baselines=self.preprocessor.baseline_input_ids,
            additional_forward_args=self.preprocessor.attention_mask,
            target=(self.preprocessor.mask_index, self.preprocessor.ground_truth_index),
        )

        attributions_word = summarize_attributions(attributions[0])
        top_words, top_words_val, top_word_ind = get_top_k_attributed_tokens(
            attributions_word, k=k, preprocessor=self.preprocessor
        )

        df = pd.DataFrame(
            {
                "Word(Index), Attribution": [
                    "{} ({}), {}".format(word, pos, round(val.item(), 2))
                    for word, pos, val in zip(top_words, top_word_ind, top_words_val)
                ]
            }
        )
        df.style.set_properties(cell_ids=False)

        full_token_list = [
            "{}({})".format(token, str(i)) for i, token in enumerate(self.preprocessor.all_tokens)
        ]

        print(f"Full token list: {full_token_list}")
        print(f"Top {k} attributed embeddings for sentiment prediction: {df}")
