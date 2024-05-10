from external_metrics import rouge
import logging


# Calculate ROUGE scores
def rouge(hypotheses, references):
    rouge_score = 0
    n_seq = len(hypotheses)

    for h, r in zip(hypotheses, references):
        rouge_score += rouge.calc_score(hypotheses=[h], references=[r]) / n_seq

    return rouge_score

def print_rouge(hypotheses, references, all=False):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    logger = logging.getLogger(__name__)

    rouge_score = rouge(hypotheses, references) * 100

    rougeStr = "Rouge: " + str(rouge_score)

    logger.info(rougeStr)
    #print(rougeStr)


