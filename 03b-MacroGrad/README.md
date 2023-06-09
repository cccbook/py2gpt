# MacroGrad -- 修改自 MicroGrad

Karpathy 的 micrograd 梯度引擎的主要類別 Value 是以單一數值為單位進行反傳遞梯度計算。

* https://github.com/newcodevelop/micrograd/

Dibyanayan Bandyopadhyay (github 名稱為 newcodevelop) 將其修改為以 numpy 的 ndarray 為單位進行計算。

* https://github.com/newcodevelop/micrograd/

這樣就可以讓原本只能對純量進行反傳遞梯度計算的 micrograd 變成可以對更高維的《向量，矩陣或張量》進行梯度計算的 macrograd 了。

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
