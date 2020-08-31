"""Config for a linear regression model evaluated on a diabetes dataset."""
from dbispipeline.evaluators import GridEvaluator
import dbispipeline.result_handlers as result_handlers
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from mediaeval2020.dataloaders.diabetes import DiabetesLoader
import mediaeval2020.evaluators as evaluators

dataloader = DiabetesLoader()

pipeline = Pipeline([
    ('model', LinearRegression()),
])

evaluator = GridEvaluator(
    parameters={},
    grid_parameters=evaluators.grid_parameters(),
)

result_handlers = [
    result_handlers.print_gridsearch_results,
]
