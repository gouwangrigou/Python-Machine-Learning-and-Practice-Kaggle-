# coding=utf-8

# 线性回归器-美国波士顿地区放假预测

# 从sklearn.datasets导入波士顿房价数据读取器。
from sklearn.datasets import load_boston

# 从读取房价数据存储在变量boston中。
boston = load_boston()
# 输出数据描述。
print(boston.DESCR)

# 从sklearn.cross_validation导入数据分割器。
from sklearn.model_selection import train_test_split

# 导入numpy并重命名为np。
import numpy as np

X = boston.data
y = boston.target

# 随机采样25%的数据构建测试样本，其余作为训练样本。
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.25)

# 分析回归目标值的差异。
print("The max target value is", np.max(boston.target))
print("The min target value is", np.min(boston.target))
print("The average target value is", np.mean(boston.target))

# 从sklearn.preprocessing导入数据标准化模块。
from sklearn.preprocessing import StandardScaler

# 分别初始化对特征和目标值的标准化器。
ss_X = StandardScaler()
ss_y = StandardScaler()

# 分别对训练和测试数据的特征以及目标值进行标准化处理。
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)

y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

# 从sklearn.linear_model导入LinearRegression。
from sklearn.linear_model import LinearRegression

# 使用默认配置初始化线性回归器LinearRegression。
lr = LinearRegression()
# 使用训练数据进行参数估计。
lr.fit(X_train, y_train)
# 对测试数据进行回归预测。
lr_y_predict = lr.predict(X_test)

# 从sklearn.linear_model导入SGDRegressor。
from sklearn.linear_model import SGDRegressor

# 使用默认配置初始化线性回归器SGDRegressor。
sgdr = SGDRegressor()
# 使用训练数据进行参数估计。
sgdr.fit(X_train, y_train)
# 对测试数据进行回归预测。
sgdr_y_predict = sgdr.predict(X_test)
# 使用LinearRegression模型自带的评估模块，并输出评估结果。
print('The value of default measurement of LinearRegression is', lr.score(X_test, y_test))

# 从sklearn.metrics依次导入r2_score、mean_squared_error以及mean_absoluate_error用于回归性能的评估。
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 使用r2_score模块，并输出评估结果。
print('The value of R-squared of LinearRegression is', r2_score(y_test, lr_y_predict))

# 使用mean_squared_error模块，并输出评估结果。
print('The mean squared error of LinearRegression is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))

# 使用mean_absolute_error模块，并输出评估结果。
print('The mean absoluate error of LinearRegression is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(lr_y_predict)))

# 使用SGDRegressor模型自带的评估模块，并输出评估结果。
print('The value of default measurement of SGDRegressor is', sgdr.score(X_test, y_test))

# 使用r2_score模块，并输出评估结果。
print('The value of R-squared of SGDRegressor is', r2_score(y_test, sgdr_y_predict))

# 使用mean_squared_error模块，并输出评估结果。
print('The mean squared error of SGDRegressor is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(sgdr_y_predict)))

# 使用mean_absolute_error模块，并输出评估结果。

# 从sklearn.svm中导入支持向量机（回归）模型。
from sklearn.svm import SVR

# 使用线性核函数配置的支持向量机进行回归训练，并且对测试样本进行预测。
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train, y_train)
linear_svr_y_predict = linear_svr.predict(X_test)

# 使用多项式核函数配置的支持向量机进行回归训练，并且对测试样本进行预测。
poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

# 使用径向基核函数配置的支持向量机进行回归训练，并且对测试样本进行预测。
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

# 使用R-squared、MSE和MAE指标对三种配置的支持向量机（回归）模型在相同测试集上进行性能评估。
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print('R-squared value of linear SVR is', linear_svr.score(X_test, y_test))
print('The mean squared error of linear SVR is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print('The mean absoluate error of linear SVR is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print('R-squared value of Poly SVR is', poly_svr.score(X_test, y_test))
print('The mean squared error of Poly SVR is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print('The mean absoluate error of Poly SVR is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print('R-squared value of RBF SVR is', rbf_svr.score(X_test, y_test))
print('The mean squared error of RBF SVR is',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print('The mean absoluate error of RBF SVR is',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))

# 从sklearn.neighbors导入KNeighborRegressor（K近邻回归器）。
from sklearn.neighbors import KNeighborsRegressor

# 初始化K近邻回归器，并且调整配置，使得预测的方式为平均回归：weights='uniform'。
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train, y_train)
uni_knr_y_predict = uni_knr.predict(X_test)

# 初始化K近邻回归器，并且调整配置，使得预测的方式为根据距离加权回归：weights='distance'。
dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train, y_train)
dis_knr_y_predict = dis_knr.predict(X_test)

# 使用R-squared、MSE以及MAE三种指标对平均回归配置的K近邻模型在测试集上进行性能评估。
print('R-squared value of uniform-weighted KNeighorRegression:', uni_knr.score(X_test, y_test))
print('The mean squared error of uniform-weighted KNeighorRegression:',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))
print('The mean absoluate error of uniform-weighted KNeighorRegression',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(uni_knr_y_predict)))

# 使用R-squared、MSE以及MAE三种指标对根据距离加权回归配置的K近邻模型在测试集上进行性能评估。
print('R-squared value of distance-weighted KNeighorRegression:', dis_knr.score(X_test, y_test))
print('The mean squared error of distance-weighted KNeighorRegression:',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))
print('The mean absoluate error of distance-weighted KNeighorRegression:',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dis_knr_y_predict)))

# 从sklearn.tree中导入DecisionTreeRegressor。
from sklearn.tree import DecisionTreeRegressor

# 使用默认配置初始化DecisionTreeRegressor。
dtr = DecisionTreeRegressor()
# 用波士顿房价的训练数据构建回归树。
dtr.fit(X_train, y_train)
# 使用默认配置的单一回归树对测试数据进行预测，并将预测值存储在变量dtr_y_predict中。
dtr_y_predict = dtr.predict(X_test)

# 使用R-squared、MSE以及MAE指标对默认配置的回归树在测试集上进行性能评估。
print('R-squared value of DecisionTreeRegressor:', dtr.score(X_test, y_test))
print('The mean squared error of DecisionTreeRegressor:',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))
print('The mean absoluate error of DecisionTreeRegressor:',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(dtr_y_predict)))

# 从sklearn.ensemble中导入RandomForestRegressor、ExtraTreesGressor以及GradientBoostingRegressor。
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

# 使用RandomForestRegressor训练模型，并对测试数据做出预测，结果存储在变量rfr_y_predict中。
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
rfr_y_predict = rfr.predict(X_test)

# 使用ExtraTreesRegressor训练模型，并对测试数据做出预测，结果存储在变量etr_y_predict中。
etr = ExtraTreesRegressor()
etr.fit(X_train, y_train)
etr_y_predict = etr.predict(X_test)

# 使用GradientBoostingRegressor训练模型，并对测试数据做出预测，结果存储在变量gbr_y_predict中。
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
gbr_y_predict = gbr.predict(X_test)

# 使用R-squared、MSE以及MAE指标对默认配置的随机回归森林在测试集上进行性能评估。
print('R-squared value of RandomForestRegressor:', rfr.score(X_test, y_test))
print('The mean squared error of RandomForestRegressor:',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))
print('The mean absoluate error of RandomForestRegressor:',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rfr_y_predict)))

# 使用R-squared、MSE以及MAE指标对默认配置的极端回归森林在测试集上进行性能评估。
print('R-squared value of ExtraTreesRegessor:', etr.score(X_test, y_test))
print('The mean squared error of  ExtraTreesRegessor:',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))
print('The mean absoluate error of ExtraTreesRegessor:',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(etr_y_predict)))

# 利用训练好的极端回归森林模型，输出每种特征对预测目标的贡献度。
print(np.sort(zip(etr.feature_importances_, boston.feature_names), axis=0))

# 使用R-squared、MSE以及MAE指标对默认配置的梯度提升回归树在测试集上进行性能评估。
print('R-squared value of GradientBoostingRegressor:', gbr.score(X_test, y_test))
print('The mean squared error of GradientBoostingRegressor:',
      mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))
print('The mean absoluate error of GradientBoostingRegressor:',
      mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(gbr_y_predict)))
