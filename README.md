# Modified FCOS

Tugas pengenalan pola

## Summary

### Res2Net

- Masalah utama yang ingin dipecahkan dari Res2Net adalah representasi fitur multiple scale yang masih menggunakan layer-wise (Bergantung besar terhadap multi-layer).
- Res2Net menghadirkan cara lain untuk multiscale feature extraction, yakni dengan hierarchical residual-like block. Block ini sama seperti ResNet namun proses di dalamnya sedikit berbeda.
- Novelty dari Res2Net adalah:
  - Res2Net module: Membagi input n-channel menjadi s-grup dengan w-channel, dimana n=s*w
  
### ShuffleNet

- Masalah utama yang ingin dipecahkan oleh penulis adalah mengecilkan komputasi sehingga bisa berjalan pada perangkat edge maupun gawai tanpa mengorbankan akurasi.
- ShuffleNet sendiri merupakan network-in-network seperti ResNet, sehingga membutuh arsitektur utama untuk diimplementasikan.
- Novelty yang diberikan dari paper ini adalah
  - Pointwise group convolution: filter 1x1 dari group convolution
  - Channel shuffle operation: melakukan pengacakan channel sebelum melakukan depth-wise separable convolution

### NasNet search space

- Masalah utama membangun sebuah arsitektur secara otomatis, jadi tidak handcrafted
- NASNet sendiri merupakan arsitektur yang dibangun menggunakan neural architecture search. Studi yang dilakukan pada NASNet ini ada pada tingkat micro, yakni dengan arsitektur yang sudah fix, NAS akan mengisi blok-blok yang sudah disediakan. NAS akan mencari dari architecture space, konfigurasi blok mana yang menghasilkan performa terbaik
- Novelty dari NASnet adalah
  - Fixed architecture base
  - Meminimalkan resource saat mencari konfigurasi blok terbaik
  - scheduledDropPath regularization

### CSPNet

- Tujuan utama : mengurangi kerja yang diperlukan dalam komputasi inference dengan memodifikasi arsitektur network.
- CSPNet merupakan Backbone yang bisa menjadi variant backbone lain (mis. CSPDenseNet sebagai varian CSP untuk DenseNet)
- Novelty yang diberikan dari paper ini adalah
  - Cross Stage Partial Network: partisi feature map menjadi dua bagian dan digabung setelah stage selesai
  - Exact Fusion model: Alternative Feature Pyramid network, namun melakukan fusion tergantung pada ukuran anchor.

### MobileNet V2

- Tujuan utama : Mendorong state of the art model computer vision untuk implementasi mobile, dengan mengurangi operasi dan memori yang dibutuhkan.
- MobileNetV2 merupakan network in network seperti ShuffleNet. Dalam paper ini dikenalkan SSDLite, varian dari SSD dengan mengimplementasikan konsep MobileNetV2 pada SSD.
- Novelty yang diberikan dari paper ini adalah
  - Linear Bottleneck: Menambahkan linear bottleneck ke dalam blok konvolusi untuk mempertahankan informasi yang mungkin hilang karena non-linearity (e.g. ReLU).
  - Inverted Residual Block: Bila biasanya residual bypass bottleneck layer, namun dengan intuisi linear bottleneck bahwa informasi banyak pada bottleneck, maka dibuat shortcut antar bottleneck layer untuk preserve informasi tersebut.

### FCOS (Fully Convolutional One-stage)

- Tujuan utama : Melakukan object detection dengan pendekatan per-pixel (seperti semantic segmentation). Dengan framework yang sederhana dan fleksibel sembari meningkatkan akurasi deteksi.
- FCOS merupakan network architecture (end to end, sepadan dengan YOLO) untuk mendeteksi objek. Dengan menggunakan Fully Convolutional Network dan Feature Pyramid Network tanpa menggunakan anchor detector.
- Novelty yang diberikan dari paper ini adalah
  - Center-ness : Metrics untuk menekan low-quality bounding box sebagai output dari arsitektur ini selain klasifikasi kelas dan regresi bounding box


### Kelemahan

- Res2Net: **Tidak membahas dasar pembagian dataset** serta dijelaskan jumlah blok tersebut direplikasi berapa banyak pada arsitektur Faster R-CNN
- ShuffleNet: **Belum ada metrik** untuk mengukur apakah informasi yang dihasilkan tersebut lebih baik atau tidak. Mungkin bisa menggunakan pengujian saliency map test sebagai auxiliary test
- NASNet Search Space: **Membutuhkan resource yang cukup besar** untuk menghasilkan satu model. Untuk saat NAS bukanlah solusi serta hasil yang didapat jauh lebih murah menggunakan handcrafted. Namun disisi lain hasil dari NASNet bisa digunakan sebagai dasar untuk pengembangan handcrafted model
- CSPNet: **Perbandingan antar faktor yang banyak** dan dengan faktor yang kurang jelas membuat cukup sulit untuk mengetahui faktor apa yang berpengaruh secara signifikan
- MobileNetV2: Penggunaan metrics kompleksitas yang berbeda (MAdds) membuat sulit untuk membandingkan dengan metode lain
- FCOS: Tidak membahas mengenai **performa komputasi** (FPS/ BFLOPS/ Memori) sehingga sulit untuk mendapat gambaran penggunaan riilnya. Beberapa data ada di github namun tidak dibandingkan dengan komprehensif terhadap metode lain.


## Desain Eksperimen

### Tujuan Penelitian 

Mengukur dan meningkatkan efisiensi komputasi FCOS dengan menggunakan Network-in-Network block pada backend-nya. Sembari mempertahankan efikasi (presisi).

### Variable Response

- Mengukur average precision 
- Jumlah Parameters
- BFLOPS dan FPS

### Faktor yang diuji

- Perbandingan penggunaan CSP, Res2Net atau ShuffleNet sebagai NIN Block
- Benchmark: FCOS, YOLO, SSD dan RetinaNet


### Skenario eksperimen
- Mengikuti setup FCOS
- Holdout Validation
- Saliency map testing (auxiliary test)

## Resources

1. [FCOS](https://github.com/tianzhi0549/FCOS)
2. [Res2net](https://github.com/Res2Net/Res2Net-PretrainedModels)
3. [ShuffleNet](https://github.com/jaxony/ShuffleNet)
4. [Presentasi](https://docs.google.com/presentation/d/15RQL2SX85Vk3ctm463_T8NDku4Ca-ljMRMCTL3KPz9g/edit?usp=sharing)
