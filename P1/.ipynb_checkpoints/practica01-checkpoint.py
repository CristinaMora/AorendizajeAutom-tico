from public_tests import *
from utils import load_data_csv

from LinearRegression import LinearReg
from LinearRegression import cost_test_obj
from LinearRegression import compute_gradient_obj


def test_gradient(x_train, y_train):
    initial_w = 0
    initial_b = 0

    lr = LinearReg(x_train,y_train,initial_w,initial_b)
    lr_tmp_dj_dw, lr_tmp_dj_db = lr.compute_gradient()
    print('Gradient at initial w, b (similar to)(-50.498559851122586 -6.988606075159082):',
          lr_tmp_dj_dw, lr_tmp_dj_db)

    test_w = 0.2
    test_b = 0.2
    
    lr = LinearReg(x_train,y_train,test_w,test_b)
    lr_tmp_dj_dw, lr_tmp_dj_db = lr.compute_gradient()
    print('Gradient at initial w, b (similar to)(-38.7005262336415 -5.369312042261976):',
          lr_tmp_dj_dw, lr_tmp_dj_db)

    print("----Compute gradient POO-------")
    compute_gradient_test(compute_gradient_obj)

def test_cost(x_train, y_train):

    initial_w = 2
    initial_b = 1

    lr = LinearReg(x_train,y_train,initial_w,initial_b)
    lrcost = lr.compute_cost()
    print(type(lrcost))
    print(f'Cost at initial w (35.841): {lrcost:.3f}')

    print("----Compute cost POO-------")
    compute_cost_test(cost_test_obj)



def test_gradient_descent(x_train, y_train, w, b):
    predict1 = 3.5 * w + b
    print('for score 3.5, we predict user score of (3.97) %.2f' %
          predict1)
    print(predict1)
    print("Case 1")
    assert np.allclose(predict1, 3.96667965 )
    print("Case 1 passed!")

    predict2 = 7.0 * w + b
    print('for score 7.9, we predict user score of (6.86) %.2f' %
          predict2)
    print("Case 2")
    print(predict2)
    assert np.allclose(predict2, 6.85867675)
    print("Case 2 passed!")
    print("\033[92mAll tests passed!")


#TO-DO the main program.