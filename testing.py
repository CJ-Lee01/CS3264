from typing import Self

import pandas


class Model:
    def train_predict(self, data_point: str, label: int) -> int:
        """
        Gets the classification of the data point, and train on it (on specific conditions)
        :param label: The true label
        :param data_point: The article to be classified
        :return: The initial predicted label
        """
        raise NotImplementedError


class TimeBasedTester:
    def __init__(self, model: Model):
        self.model = model
        self.__text = None
        self.__truth = None
        self.__time = None
        self.predictions = []

    def with_data(self, data: pandas.DataFrame, *, time_col: str, text_col: str = "text",
                  truth_col: str = "label") -> Self:
        """
        Loads the data into the tester
        :param data: The full dataframe including the label and time information.
        :param time_col: The name of the time column.
        :param text_col: The name of the text column. Default: "text"
        :param truth_col: The name of the ground truth label. Default: "label"
        :return: Self for chaining
        """
        df = data.sort_values(by=time_col)
        self.__time = df[time_col]
        self.__text = df[text_col]
        self.__truth = df[truth_col]
        return self

    def test(self, time_cutoff=None) -> Self:
        """
        Runs the testing algorithm against
        :param time_cutoff: Time to stop the testing. Default: None. Also, requires compatibility with the time in the data
        :return: Self for chaining
        """
        for text, label, ctime in zip(self.__text, self.__truth, self.__time):
            if time_cutoff is not None and ctime > time_cutoff:
                return
            pred = self.model.train_predict(text, label)
            self.predictions.append(pred)

        return self

    def evaluate(self, *, running_start: int = 0, **eval_args):
        """
        Evaluates the function given a running start and a set of evaluation metrics
        :param running_start: Number of predictions to ignore at the start.
        :param eval_args: The evaluation functions which take in (truth, prediction). Must be named args
        :return: None
        """
        compared_truths = self.__truth[running_start:len(self.predictions)]
        compared_pred = self.predictions[running_start:]
        for eval_name in eval_args:
            print(eval_name, eval_args[eval_name](compared_truths, compared_pred))

