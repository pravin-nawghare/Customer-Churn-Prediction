
from churn_prediction.exception import CustomerChurnException
import sys



try:
    a = 2/0
except Exception as e:
    raise CustomerChurnException(e,sys)