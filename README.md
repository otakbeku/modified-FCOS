# Modified FCOS

Tugas pengenalan pola

## Tujuan Penelitian 

Mengukur dan meningkatkan efisiensi komputasi FCOS dengan menggunakan Network-in-Network block pada backend-nya. Sembari mempertahankan efikasi (presisi).

## Variable Response

- Mengukur average precision 
- Jumlah Parameters
- BFLOPS dan FPS

## Faktor yang diuji

- Perbandingan penggunaan CSP, Res2Net atau ShuffleNet sebagai NIN Block
- Benchmark: FCOS, YOLO, SSD dan RetinaNet


## Skenario eksperimen
- Mengikuti setup FCOS
- Holdout Validation
- Saliency map testing (auxiliary test)

## Resources

1. [FCOS](https://github.com/tianzhi0549/FCOS)
2. [Res2net](https://github.com/Res2Net/Res2Net-PretrainedModels)
3. [ShuffleNet](https://github.com/jaxony/ShuffleNet)
4. [Presentasi](https://docs.google.com/presentation/d/15RQL2SX85Vk3ctm463_T8NDku4Ca-ljMRMCTL3KPz9g/edit?usp=sharing)
