import numpy as np
import ray
import plotly.graph_objs as go


class ExpectedVsSampleUpdates:

    @staticmethod
    @ray.remote
    def q_updates(b: int, distribution: str) -> np.array:
        dist = getattr(np.random, distribution)(size=b)
        mu, sample_mean = np.mean(dist), 0
        rmse = np.ones(2 * b)  # the error in the initial estimate is 1

        for sample_size in range(1, 2 * b):
            successor_value = np.random.choice(dist)
            sample_mean += (successor_value - sample_mean) / sample_size
            rmse[sample_size] = np.sqrt(np.power(sample_mean - mu, 2) / sample_size)

        return rmse

    def plot_rmse(self, rmse):
        traces = list()
        for b, error in rmse.items():
            traces.append(go.Scatter(
                mode='lines',
                y=error,
                x=(np.arange(len(error)) + 1) / b,
                name=f'branching factor of {b}',
            ))
        traces.append(go.Scatter(
            mode='lines',
            y=[1, 1, 0, 0],
            x=[0, 1, 1.000001, 2],
            name=f'expected updates',
        ))
        fig = {'data': traces}
        return fig

    @staticmethod
    def description():
        description = """
        ### Reproducing Figure 8.7
        
        ---
        
        The graph shows the estimation error as a function of computation time for expected and sample
        updates for a variety of branching factors, b. The case considered is that in which all
        b successor states are equally likely and in which the error in the initial estimate is 1. 
        The values at the next states are assumed correct, so the expected update reduces
        the error to zero upon its completion.
    
        The key observation is that for moderately large b the error falls dramatically with a tiny fraction of b updates. 
        For these cases, many state–action pairs could have their values improved dramatically, to within a few
        percent of the effect of an expected update, in the same time that a single state–action
        pair could undergo an expected update.
        
        ---
        
        Exercise 8.6 The analysis above assumed that all of the b possible next states were
        equally likely to occur. Suppose instead that the distribution was highly skewed, that
        some of the b states were much more likely to occur than most. Would this strengthen or
        weaken the case for sample updates over expected updates? Support your answer.
        
        ---
        
        """
        return description
