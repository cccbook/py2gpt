# MacroGrad -- 修改自 MicroGrad

改用 numpy 去處理向量的加減乘除等反傳遞功能，這樣就可以讓
原本只能對純量進行反傳遞梯度計算的 micrograd 變成可以對向量
進行梯度計算的 macrograd 了。

## 執行 mnist.py 範例

這個範例用 MLP 多層感知器 (沒有用卷積神經網路 CNN) 來學習 MNIST 手寫數字辨識的功能。

```
$ python mnist.py
loss in step 0 is 14.908012820799556
loss in step 1000 is 0.6853565532111151
loss in step 2000 is 0.5844204554622346
loss in step 3000 is 0.47859919276689794
loss in step 4000 is 0.45641239933087924
loss in step 5000 is 0.4128653980878078
loss in step 6000 is 0.40506020894545897
loss in step 7000 is 0.3955104147867347
loss in step 8000 is 0.3774670342827101
loss in step 9000 is 0.37506717341305984
loss in step 10000 is 0.3839270105961265
loss in step 11000 is 0.35121356657827174
loss in step 12000 is 0.35536878295305846
loss in step 13000 is 0.33581069097395505
loss in step 14000 is 0.33362058549120954
loss in step 15000 is 0.32517694623935106
loss in step 16000 is 0.3320893960926535
loss in step 17000 is 0.31746543394838694
loss in step 18000 is 0.3171616065938816
loss in step 19000 is 0.3093455466843886
loss in final step 20000 is 0.2998612290886142
```
