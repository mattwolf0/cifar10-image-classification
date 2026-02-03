# CIFAR‑10 képosztályozás PyTorch-ban – SmallCNN vs MediumCNN

A célom egyszerű volt: **minél jobb teszt pontosságot** kihozni CIFAR‑10-en (10 osztály, 32×32 RGB képek).

Először egy **SmallCNN**-nel próbálkoztam, ami jól működött, de nem volt elég hatékony. Ezzel esélytelen volt megközelíteni a 90%-os pontosságot (ez volt a minimális személyes elvárásom), ezért áttértem **MediumCNN**-re, és ott már kifejezetten jó eredményeket kaptam.

---

## Futtatás

Telepítés:
```bash
pip install -r requirements.txt
```

Futtatás:
```bash
python main.py
```

A CIFAR‑10 automatikusan letöltődött a `./data` mappába.

---

## Projekt felépítése

- `main.py`: adatok betöltése, transzformok, DataLoader, tanítás/teszt, naplózás, mintaképek
- `models.py`: MediumCNN definíció
- `requirements.txt`: függőségek

---

## Mi történik futás közben? (a kód fő lépései)

### 1) Fix seed (`set_seed`)
A seedet beállítottam, hogy újrafuttatásnál az eredmény könnyen reprodukálható legyen (shuffle, random crop/flip, stb.).

### 2) Adatok és transzformok (`get_loaders`)
A `get_loaders()` feladata az volt, hogy összerakja a train/test adatpipeline-t.

- **Train**: a képeket megvariálta (augmentáció), hogy jobban általánosítson a modell.
- **Test**: nem használt augmentációt, csak a szükséges átalakítást és normalizálást (így volt fair a mérés).

Fontos: a transzformot nem én hívtam meg külön, hanem a datasethez volt kötve (`transform=tf_train`), ezért **minden kép betöltésekor automatikusan lefutott**, amikor a DataLoader batch-et készített.

**DataLoader worker-ek (`num_workers`)**: a worker-ek a háttérben készítették a batch-eket (kép betöltés + transzformok), így a tanítás kevésbé várt az adatra.

### 3) Modell
A modell két részből áll:
- `features`: a képből feature map-eket/jellemzőket csinál
- `head`: ezekből számolja ki a 10 osztály pontszámát (logits)

### 4) Loss + optimizer + scheduler
- `CrossEntropyLoss`-t használtam 10 osztályos klasszifikációhoz.
- `SGD + momentum (+ nesterov)` van beállítva, ami CNN-eknél jól bevált.
- `OneCycleLR` batchenként állítja a learning rate-et; a tréning vége felé sokat segített a finomításon.
- CUDA esetén AMP-t is használtam (mixed precision), ami gyorsította a tréninget.

### 5) Egy epoch menete (`run_epoch`)
Train módban ez történik: forward → loss → backward → súlyfrissítés → scheduler léptetés.  
Test módban csak forward és metrikaszámítás megy (`no_grad`).

A kód elmenteti a legjobb teszt eredményű állapotot (`best_state`), és a végén vissza is tölti.

### 6) Mintaképek (`show_samples`)
A futás végén pár tesztképet kirajzoltam predikcióval és a helyes címkével, hogy legyen gyors “szemre” ellenőrzés is.

---

## Adatelőkészítés (röviden)

Normalizáláshoz ezeket használtam:
- `MEAN = (0.4914, 0.4822, 0.4465)`
- `STD  = (0.2470, 0.2435, 0.2616)`

Train transzformok:
- `RandomCrop(32, padding=4)`
- `RandomHorizontalFlip()`
- `Normalize(MEAN, STD)`
- **Medium futásban:** `RandomErasing(p=0.25, ...)`

Test transzformok:
- `ToTensor()` + `Normalize(MEAN, STD)` (augmentáció nélkül)

---

## Modellek

### SmallCNN
Egy egyszerű 3 szintes CNN-t használtam (konvolúció + pooling), majd klasszikus `Flatten + Linear` osztályozót.

### MediumCNN
Egy mélyebb CNN-t használtam `Conv2d + BatchNorm2d + ReLU` blokkokkal, `Dropout2d`-vel, és egy egyszerű head-del:
`AdaptiveAvgPool2d(1) → Flatten → Linear(256→10)`.

---

## Eredmények (35 epoch)

### SmallCNN

- best **test loss**: **0.574** (epoch 30)
- epoch 35: `train_acc=0.758`, `test_loss=0.618`
- utolsó 5 epoch test loss: **0.591 ± 0.016**

A baseline javult, de lassan, és kb. epoch 30 körül érte el a legjobb teszt loss-t.

### MediumCNN

- **90% feletti teszt pontosságot** először: epoch **32**
- best **test_acc**: **0.919** (epoch 34)
- best **test loss**: **0.247** (epoch 34)
- epoch 35: `train_acc=0.931`, `test_acc=0.919`, `test_loss=0.247`
- utolsó 5 epoch test acc: **0.911 ± 0.009**

---

## Konklúzió (Small vs Medium)

A MediumCNN-nél bejött az elképzelésem: **gyorsabban** jutottam el magas pontosságig, és **magasabb plafont** is elértem.

A különbség szerintem több dologból állt össze:
- **Nagyobb modellkapacitás** (több csatorna, több konvolúció) → több mintázatot tudott megtanulni.
- **BatchNorm** → stabilabb lett a tanítás, kevésbé “szaladt el” a tréning.
- **Erősebb regulárizáció/augmentáció** (Dropout2d, RandomErasing) → jobb lett a teszt generalizáció.
- **OneCycleLR + SGD momentum** → a tréning végén jól finomította a súlyokat.

A SmallCNN baseline-ként hasznos volt (gyorsan kiderült, hogy a pipeline rendben van), de a végső célhoz (90% fölötti teszt pontosság) a MediumCNN + erősebb tréning recept kellett.

---

## Korlátok

- Egy seed-en teszteltem mindent (42). Több seed átlagolása stabilabb összehasonlítást adhatna, de ezen projekthez valószínűleg nem tenne hozzá sokat.

---

## Teljes tréning logok

### SmallCNN log
```text
epoch 01 train_acc=0.363 loss=1.714/1.343
epoch 02 train_acc=0.485 loss=1.402/1.175
epoch 03 train_acc=0.544 loss=1.266/1.034
epoch 04 train_acc=0.584 loss=1.159/0.935
epoch 05 train_acc=0.614 loss=1.088/0.886
epoch 06 train_acc=0.637 loss=1.030/0.850
epoch 07 train_acc=0.653 loss=0.984/0.808
epoch 08 train_acc=0.663 loss=0.961/0.786
epoch 09 train_acc=0.673 loss=0.934/0.769
epoch 10 train_acc=0.682 loss=0.912/0.740
epoch 11 train_acc=0.689 loss=0.891/0.702
epoch 12 train_acc=0.696 loss=0.871/0.719
epoch 13 train_acc=0.700 loss=0.862/0.707
epoch 14 train_acc=0.710 loss=0.842/0.669
epoch 15 train_acc=0.712 loss=0.831/0.679
epoch 16 train_acc=0.715 loss=0.821/0.676
epoch 17 train_acc=0.720 loss=0.810/0.639
epoch 18 train_acc=0.724 loss=0.799/0.667
epoch 19 train_acc=0.727 loss=0.789/0.655
epoch 20 train_acc=0.730 loss=0.775/0.628
epoch 21 train_acc=0.733 loss=0.773/0.620
epoch 22 train_acc=0.735 loss=0.767/0.668
epoch 23 train_acc=0.736 loss=0.761/0.604
epoch 24 train_acc=0.741 loss=0.752/0.615
epoch 25 train_acc=0.741 loss=0.750/0.606
epoch 26 train_acc=0.743 loss=0.746/0.626
epoch 27 train_acc=0.747 loss=0.735/0.645
epoch 28 train_acc=0.746 loss=0.738/0.599
epoch 29 train_acc=0.750 loss=0.726/0.594
epoch 30 train_acc=0.748 loss=0.727/0.574
epoch 31 train_acc=0.756 loss=0.718/0.593
epoch 32 train_acc=0.754 loss=0.714/0.586
epoch 33 train_acc=0.756 loss=0.711/0.574
epoch 34 train_acc=0.755 loss=0.709/0.585
epoch 35 train_acc=0.758 loss=0.701/0.618
```

### MediumCNN log
```text
epoch 01 train_acc=0.404 test_acc=0.529 loss=1.607/1.238
epoch 02 train_acc=0.567 test_acc=0.630 loss=1.209/1.025
epoch 03 train_acc=0.635 test_acc=0.696 loss=1.032/0.882
epoch 04 train_acc=0.684 test_acc=0.734 loss=0.910/0.750
epoch 05 train_acc=0.714 test_acc=0.706 loss=0.842/0.851
epoch 06 train_acc=0.729 test_acc=0.705 loss=0.797/0.929
epoch 07 train_acc=0.738 test_acc=0.658 loss=0.769/1.165
epoch 08 train_acc=0.747 test_acc=0.653 loss=0.743/1.071
epoch 09 train_acc=0.755 test_acc=0.779 loss=0.724/0.653
epoch 10 train_acc=0.762 test_acc=0.763 loss=0.703/0.722
epoch 11 train_acc=0.766 test_acc=0.746 loss=0.693/0.791
epoch 12 train_acc=0.770 test_acc=0.713 loss=0.678/0.903
epoch 13 train_acc=0.772 test_acc=0.782 loss=0.672/0.659
epoch 14 train_acc=0.778 test_acc=0.726 loss=0.659/0.806
epoch 15 train_acc=0.782 test_acc=0.799 loss=0.649/0.605
epoch 16 train_acc=0.785 test_acc=0.810 loss=0.639/0.570
epoch 17 train_acc=0.789 test_acc=0.797 loss=0.629/0.615
epoch 18 train_acc=0.791 test_acc=0.803 loss=0.620/0.579
epoch 19 train_acc=0.796 test_acc=0.788 loss=0.607/0.618
epoch 20 train_acc=0.802 test_acc=0.821 loss=0.586/0.523
epoch 21 train_acc=0.806 test_acc=0.837 loss=0.579/0.477
epoch 22 train_acc=0.810 test_acc=0.831 loss=0.564/0.506
epoch 23 train_acc=0.815 test_acc=0.816 loss=0.547/0.553
epoch 24 train_acc=0.819 test_acc=0.838 loss=0.534/0.481
epoch 25 train_acc=0.829 test_acc=0.849 loss=0.508/0.457
epoch 26 train_acc=0.836 test_acc=0.846 loss=0.489/0.473
epoch 27 train_acc=0.843 test_acc=0.877 loss=0.463/0.371
epoch 28 train_acc=0.857 test_acc=0.874 loss=0.427/0.383
epoch 29 train_acc=0.864 test_acc=0.882 loss=0.399/0.347
epoch 30 train_acc=0.878 test_acc=0.891 loss=0.362/0.329
epoch 31 train_acc=0.891 test_acc=0.898 loss=0.323/0.319
epoch 32 train_acc=0.905 test_acc=0.907 loss=0.285/0.279
epoch 33 train_acc=0.916 test_acc=0.912 loss=0.249/0.267
epoch 34 train_acc=0.926 test_acc=0.919 loss=0.219/0.247
epoch 35 train_acc=0.931 test_acc=0.919 loss=0.207/0.247
```
