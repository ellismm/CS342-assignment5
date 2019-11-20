from .models import LanguageModel, AdjacentLanguageModel, Bigram, load_model
from . import utils
import torch


def log_likelihood(model: LanguageModel, some_text: str):
    """
    Your code here

    Evaluate the log-likelihood of a given string.

    Hint: utils.one_hot might come in handy

    :param model: A LanguageModel
    :param some_text:
    :return: float
    """
    likelihood = 0.0
    x = model.predict_all(some_text)
    r = utils.one_hot(some_text)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            item = r[i][j]
            if(item == 1):
                likelihood += x[i][j]
    return likelihood
    raise NotImplementedError('log_likelihood')


def sample_random(model: LanguageModel, max_length: int = 100):
    """
    Your code here.

    Sample a random sentence from the language model.
    Terminate once you reach a period '.'

    :param model: A LanguageModel
    :param max_length: The maximum sentence length
    :return: A string
    """
    # text = some
    # x = model.predict_all("")
    # print(x)
    # rad
    text = ""
    # for()

    # for i in range(x.shape[0]):
    #         item = x[i]

    #         # print(x.shape[0])
    #         if(item > likliest):
    #             likliest = item
    #             index = i

    # text += utils.vocab[index]
    while(len(text) < max_length):
        index = 0;
        x = model.predict_all(text)[:,-1]

        index = torch.distributions.Categorical(logits=x).sample()

        text += utils.vocab[index]
        if text[len(text) - 1] == '.':
            break
    return text
    raise NotImplementedError('sample_random')


class TopNHeap:
    """
    A heap that keeps the top N elements around
    h = TopNHeap(2)
    h.add(1)
    h.add(2)
    h.add(3)
    h.add(0)
    print(h.elements)
    > [2,3]

    """
    def __init__(self, N):
        self.elements = []
        self.N = N

    def add(self, e):
        from heapq import heappush, heapreplace
        if len(self.elements) < self.N:
            heappush(self.elements, e)
        elif self.elements[0] < e:
            heapreplace(self.elements, e)


def beam_search(model: LanguageModel, beam_size: int, n_results: int = 10, 
    max_length: int = 100, average_log_likelihood: bool = False):
    """
    Your code here

    Use beam search for find the highest likelihood generations, such that:
      * No two returned sentences are the same
      * the `log_likelihood` of each returned sentence is as large as possible

    :param model: A LanguageModel
    :param beam_size: The size of the beam in beam search (number of sentences to keep around)
    :param n_results: The number of results to return
    :param max_length: The maximum sentence length
    :param average_log_likelihood: Pick the best beams according to the average log-likelihood, not the sum
                                   This option favors longer strings.
    :return: A list of strings of size n_results
    """
    def findLikelihood(m: LanguageModel, some_text: str):
        likelihood = []
        x = model.predict_all(some_text)
        r = utils.one_hot(some_text)
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                item = r[i][j]
                if(item == 1):
                    likelihood.append(x[i][j])
        return likelihood.mean()


    beam = TopNHeap(beam_size)

    print(beam_size)
    print(average_log_likelihood)
    result = []
    for i in range(n_results):
        result.append(sample_random(model, max_length))
    return result

    raise NotImplementedError('beam_search')


if __name__ == "__main__":
    """
      Some test code.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', choices=['Adjacent', 'Bigram', 'TCN'], default='Adjacent')
    args = parser.parse_args()

    lm = AdjacentLanguageModel() if args.model == 'Adjacent' else (load_model() if args.model == 'TCN' else Bigram())

    for s in ['abcdefg', 'abcgdef', 'abcbabc', '.abcdef', 'fedcba.']:
        print(s, float(log_likelihood(lm, s)))
    print()

    for i in range(10):
        s = sample_random(lm)
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100):
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100, average_log_likelihood=True):
        print(s, float(log_likelihood(lm, s)) / len(s))
