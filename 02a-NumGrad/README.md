# 專案 -- NumGrad

> 梯度下降法實作，用數值的方法計算梯度

關於梯度下降法的原理，請參考 -- https://github.com/cccbook/py2gpt/wiki/grad

## NumGrad 應用範例

有了 NumGrad 之後，我們就可以用來尋找函數的最小值，以下是兩個範例：

1. [gdArray.py](gdArray.py): 尋找函數  $f(x,y,z) = (x-1)^2 + (y-2)^2 + (z-3)^2$ 的最小值
    * [gdArray.py -- 用梯度下降法尋找 f(x,y,z) 函數的最小值](gdArray.md)
2. [gdRegression.py](gdRegression.py): 尋找一迴歸直線，讓直線和樣本間的誤差平方和最小 (最小平方法)。
    * [gdRegression.py -- 用梯度下降法尋找迴歸線](gdRegression.md)

