### 1. Budget gate hiện tại không thể chấp nhận bất kỳ scene nào

Trong Supplementary:

* Dòng 491–505:
  [
  \widehat d_{\mathrm{anch}}^{\mathrm{pre}}
  =========================================

  \max \widehat r_{\mathrm{cell}}
  +r_{\mathrm{noise}}
  +\lambda_{\mathrm{env}}^\top u_{\mathrm{env}}.
  ]
* Dòng 522–523 cho:
  [
  r_{\mathrm{noise}}=2.93\text{ m}
  \quad\text{hoặc}\quad4.18\text{ m}.
  ]
* Dòng 653–658 yêu cầu:
  [
  \widehat d_{\mathrm{anch}}^{\mathrm{pre}}\le R_{\mathrm{basin}}.
  ]
* Dòng 2100 đặt:
  [
  R_{\mathrm{basin}}=2.50\text{ m}.
  ]

Theo chính định nghĩa trên:

[
\widehat d_{\mathrm{anch}}^{\mathrm{pre}}
\ge r_{\mathrm{noise}}.
]

Do đó:

* nuScenes: (\widehat d_{\mathrm{anch}}^{\mathrm{pre}}\ge2.93>2.50);
* CARLA/Waymo: (\widehat d_{\mathrm{anch}}^{\mathrm{pre}}\ge4.18>2.50).

Như vậy (\mathcal S(x)=0) cho mọi scene, đồng nghĩa hệ thống phải fallback 100%, trái với kết quả fallback 5.0%.

Đây là **lỗi nội tại có thể làm mất tính hợp lệ của toàn bộ adaptive budget result**.

Ngoài ra, các phân vị Gaussian đang ghi cũng sai. Với định nghĩa “95% quantile of the Euclidean norm of a (D)-dimensional standard Gaussian”, giá trị đúng là:

[
\chi_{12,0.95}\approx4.585,\qquad
\chi_{24,0.95}\approx6.034,
]

không phải (3.50) và (4.99). Với (\sqrt{0.7}=0.837):

[
r_{\mathrm{noise}}\approx3.84\text{ m}\quad(D=12),
]
[
r_{\mathrm{noise}}\approx5.05\text{ m}\quad(D=24).
]

Điều này làm mâu thuẫn với (R_{\mathrm{basin}}=2.50) còn nghiêm trọng hơn.

**Cách sửa:** phải xác định lại norm thực sự được dùng. Có khả năng tác giả định dùng trajectory RMS, tức chia cho (\sqrt D), thay vì full-vector Euclidean norm. Sau khi sửa định nghĩa phải:

1. tính lại (r_{\mathrm{noise}});
2. hiệu chỉnh lại (R_{\mathrm{basin}});
3. chạy lại budget admission;
4. cập nhật fallback distribution và các bảng CARLA.

Không nên chỉ tăng (R_{\mathrm{basin}}) trên giấy mà không chạy lại kết quả.

---

### 2. Hiệu chỉnh Bradley–Terry đang áp dụng lên một đại lượng không được định danh

Supplementary dòng 975–993 huấn luyện Bradley–Terry bằng chênh lệch:

[
s_{ij}=
\frac{r_\phi(o,\tau_i)-r_\phi(o,\tau_j)}
{\beta_{\mathrm{BT}}}.
]

Loss này chỉ xác định **chênh lệch utility**. Với bất kỳ hàm (c(o)):

[
r'*\phi(o,\tau)=r*\phi(o,\tau)+c(o)
]

vẫn cho cùng loss và cùng pairwise probability. Do đó giá trị tuyệt đối (r_\phi(o,\tau_k)) không được định danh.

Nhưng tại Supplementary dòng 691–707 và Submission dòng 413, runtime lại tính:

[
p_k^{\mathrm{cal}}
==================

C_{\mathrm{iso}}!\left(r_\phi(o,\tau_k)\right),
]

rồi chuyển thành log-odds. Đây không phải đại lượng pairwise đã được mô hình Bradley–Terry học.

Ngoài ra:

* Eq. isotonic calibration tại Supplementary dòng 1076–1089 dùng (s_n), nhưng (s_n) không được định nghĩa rõ là raw utility hay pairwise margin.
* ECE/Brier được báo cáo trên 6000 **pairs**, trong khi runtime sử dụng probability cho một **candidate đơn lẻ**.

Vì vậy ECE (=0.060), Brier (=0.120) chưa chứng minh rằng probability đang dùng trong runtime được hiệu chỉnh đúng.

**Cách sửa hợp lệ:**

* hiệu chỉnh pairwise margin (r_i-r_j), rồi dùng selector pairwise/tournament;
* hoặc định nghĩa một reference trajectory cố định (\tau_{\mathrm{ref}}) và hiệu chỉnh
  [
  r(o,\tau_k)-r(o,\tau_{\mathrm{ref}});
  ]
* hoặc center utility theo candidate set, chẳng hạn
  [
  \bar r_k=r_k-\frac1K\sum_j r_j,
  ]
  sau đó huấn luyện và hiệu chỉnh đúng đại lượng centered đó.

Sau khi thay đổi phải tính lại ECE, Brier, NLL, preference thresholds và các kết quả selector/CARLA liên quan.

---

### 3. (U_{\mathrm{pref}}^{\mathrm{pre}}) được dùng nhưng hoàn toàn chưa được định nghĩa

Đại lượng này xuất hiện trong pre-run scene gate tại:

* Submission dòng 341, 361, 385;
* Supplementary dòng 658, 678;
* threshold (U_{\mathrm{pref}}^{\mathrm{pre}}\le0.28) tại Supplementary dòng 2107.

Tuy nhiên trong toàn bộ hai file, không có công thức hoặc mô tả rõ:

* input của head này là gì;
* được tính thế nào trước khi sinh candidate;
* nhãn huấn luyện là gì;
* loss nào áp dụng;
* ECE/Brier của pre-gate;
* quan hệ giữa (U_{\mathrm{pref}}^{\mathrm{pre}}) và post-candidate (U_{\mathrm{pref}}(\widetilde\tau_k)).

Algorithm runtime tại Supplementary dòng 1163 cũng chỉ viết chung là “estimate uncertainty gates”.

Đây là biến nằm trực tiếp trong điều kiện chọn (N), nên không thể để undefined.

Ngoài ra, bảng threshold có hard gate riêng:

[
u_{\mathrm{anch}}\le0.40,
]

nhưng Eq. scene admission không chứa điều kiện này. (u_{\mathrm{anch}}) chỉ đi vào một weighted sum trong (\widehat d_{\mathrm{anch}}^{\mathrm{pre}}), không tương đương với hard threshold (0.40).

---

### 4. Mô hình được mô tả là “6-step student” nhưng deployment chạy (N={4,6,8,10,12})

Các vị trí mâu thuẫn:

* Supplementary dòng 333: “Progressive consistency distillation trains a 6-step student”.
* Supplementary dòng 1222: distillation (50\rightarrow25\rightarrow12\rightarrow6).
* Submission dòng 170 và Supplementary dòng 667: deployment chọn (N\in{4,6,8,10,12}).
* Algorithm chạy cùng “DDIM student” với (N_{\mathrm{run}}) bước.

Bài chưa cho biết:

* một checkpoint 6-step có được chạy với 4, 8, 10, 12 bước không;
* có checkpoint riêng cho từng (N) hay không;
* timestep grid cho từng (N);
* các budget 4, 8, 10 có xuất hiện trong training hay consistency distillation không;
* (g_{\psi,i,N}) được fit trên output của checkpoint nào.

Đây là khoảng trống tái lập nghiêm trọng. Với progressive distillation thông thường, không thể mặc nhiên coi checkpoint cuối cùng là một arbitrary-step sampler.

Cần chỉ rõ một trong hai trường hợp:

1. **Một variable-step model:** mô tả training schedule, timestep parameterization và validation cho từng (N).
2. **Nhiều checkpoint:** ghi checkpoint/student tương ứng cho (N=4,6,8,10,12), chi phí lưu trữ và cách chọn model.

Nếu hiện tại chỉ có student 6-step, không nên báo kết quả “VINAD-Deployed adaptive (\mathcal N)” cho tới khi cơ chế trên được thực nghiệm đầy đủ.

---

### 5. “70.5% upper bound” là sai về xác suất

Submission dòng 411 và Supplementary dòng 1659, 1665 viết:

[
0.86\times0.82\approx70.5%
]

và gọi đây là “BT-vs-majority-human upper bound”.

Điều này không đúng:

* tích trên không phải upper bound;
* cũng không phải BT–human agreement dưới giả thiết độc lập, vì trường hợp BT và human đều bất đồng với VLM vẫn có thể khiến BT và human đồng ý nhau.

Chỉ từ:

[
P(\mathrm{BT}=\mathrm{VLM})=0.86,\qquad
P(\mathrm{VLM}=\mathrm{Human})=0.82,
]

BT–human agreement có thể nằm trong khoảng:

[
0.68\le
P(\mathrm{BT}=\mathrm{Human})
\le0.96.
]

Do đó phải:

* xóa hàng “upper bound”;
* hoặc trực tiếp đánh giá BT với majority-human trên 500 pairs;
* hoặc chỉ nói đây là “product of two agreement rates”, không diễn giải thành accuracy hay bound.

Đây là lỗi reviewer về statistics có thể nhận ra rất nhanh.

---

### 6. Adaptive latency narrative đang tự mâu thuẫn

Submission dòng 330–331 nói rõ deployed headline chỉ dùng quality admission và **không dùng latency gate**.

Nhưng caption Fig. budget selection tại dòng 367 lại viết:

> “the latency gate jointly determine[s] the admissible budget set”.

Trong khi đó dòng 632–635 cho biết:

* (N\ge8) vượt 28 ms;
* timing (N=12) chỉ là linear projection;
* adaptive deployment p50/p99 chưa được đo;
* measurement được để lại cho camera-ready.

Do đó cần:

* xóa latency gate khỏi flowchart/caption của deployed headline; hoặc
* thực sự đưa latency vào (\mathcal B_{\mathrm{time}}) và đo timing cho từng (N), đặc biệt (N=12).

Câu “left to camera-ready measurement” cũng không nên giữ. Reviewer đánh giá phiên bản hiện tại, không nên yêu cầu họ tin vào kết quả sẽ được đo sau khi chấp nhận.

---

## II. Các lỗi bắt buộc sửa trong source và cấu trúc

### 7. Có một section Introduction rỗng ngay trước bibliography

`Submission.tex`, dòng 971:

```latex
\section{Introduction}
```

Section này xuất hiện lần thứ hai, ngay sau Conclusion và trước References. Phải xóa hoàn toàn. Nếu giữ, PDF sẽ có một section đánh số rỗng và phá cấu trúc bài.

---

### 8. Ảnh biography của tác giả đầu tiên đang dùng nhầm hình tổng quan bài báo

`Submission.tex`, dòng 980:

```latex
{image/fig1_vinad_overview_drawn.pdf}
```

được dùng làm portrait của Tran Duc Anh. Đây rõ ràng là nhầm file. Phải thay bằng ảnh chân dung hoặc bỏ tùy chọn ảnh.

---

### 9. Main file hiện không thể biên dịch chỉ với bộ file đã gửi

Input bắt buộc tại dòng 644:

```latex
\input{tables/per_n_latency_projection}
```

không có trong bộ file.

Các hình của main paper và hai hình của Supplementary cũng không được cung cấp. Vì vậy ba file hiện tại chưa phải một submission archive có thể biên dịch độc lập.

Các lỗi đóng gói đáng chú ý:

* thiếu `tables/per_n_latency_projection.tex`;
* thiếu toàn bộ graphics;
* thiếu các ảnh biography;
* `\doi{}` đang để rỗng tại dòng 106;
* Supplementary có duplicate hyperlink destinations `ALG@line.1` đến `ALG@line.13`;
* Supplementary có hai overfull boxes khoảng 3.6 pt và 7.6 pt.

Tôi đã thử compile Supplementary ở chế độ draft graphics: tài liệu tạo được PDF 28 trang, nên không thấy lỗi LaTeX fatal trong phần nội dung chính. Main paper chưa thể được xác nhận compile đầy đủ vì thiếu `ieeeaccess.cls`, bảng input và graphics trong bộ file cung cấp.

---

### 10. Hard-coded “Appendix M” đang trỏ sai

Submission dòng 328 viết:

> “Appendix M preserves the Langevin-flow diagnostic…”

Theo thứ tự appendix hiện tại:

* Appendix J: Local Reverse-Update Diagnostic;
* Appendix K: Non-Deployed Diagnostic Derivation and Error Accumulation;
* Appendix M: Step Pareto and Anchor Evidence.

Vì vậy Appendix M là sai. Nên dùng tham chiếu động, chẳng hạn:

```latex
Appendix~\ref{sec:app_theory_proofs}
```

thay vì ghi cứng ký tự appendix.

---

## III. Lỗi nghiêm trọng trong `refs.bib`

### 11. Entry NAVSIM có danh sách tác giả sai

Entry `dauner2024navsim`, dòng 146–151, hiện ghi sáu tác giả, trong đó có Velat Zelič, Philipp Stoll và Zehao Zhang. Danh sách này không khớp bài NAVSIM chính thức.

Bản chính thức có 12 tác giả, bắt đầu bằng Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng… và được xuất bản tại NeurIPS 2024 Datasets and Benchmarks Track. ([NeurIPS Proceedings][1])

Đây là lỗi bibliographic metadata bắt buộc sửa.

### 12. Entry SparseDrive sai tên tác giả đầu và tiêu đề

Entry `sun2024sparsedrive` đang ghi:

```bibtex
author={Sun, Wenyuan and others}
title={... with sparse scene representations}
```

Trong bản chính thức:

* tác giả đầu là **Wenchao Sun**, không phải Wenyuan Sun;
* tiêu đề là **“SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation”**;
* danh sách tác giả gồm Wenchao Sun, Xuewu Lin, Yining Shi, Chuang Zhang, Haoran Wu và Sifa Zheng. ([arXiv][2])

---

## IV. Những điểm quan trọng khác nên sửa

### 13. Thủ tục “conformal slack” chưa đủ để đưa ra ý nghĩa conformal

Supplementary dòng 618–646 chỉ dùng empirical (1-\delta) quantile của route maxima nhưng không nêu:

* số route calibration;
* finite-sample quantile correction;
* exchangeability unit;
* coverage target;
* empirical route-level coverage.

Bài đã thận trọng gọi đây là “calibrated rejection score”, điều đó tốt. Tuy nhiên nếu vẫn dùng từ “conformal”, cần định nghĩa order statistic chính xác và báo coverage; nếu không, nên gọi là “route-cluster residual quantile slack”.

### 14. Volumetric covering bound có khả năng viết sai chiều

Supplementary dòng 432 viết:

> “a standard volumetric covering bound requires at least ((3/\eta)^D) probes”.

Các covering-number expression dạng ((1+2/\eta)^D) thường được dùng như existence upper bound cho một net, không thể tự động chuyển thành lower bound “requires at least”. Nên trích theorem chính xác hoặc viết an toàn hơn:

> “the covering number scales exponentially with (D)”.

Phần này không vận hành trong runtime nên không phá kết quả chính, nhưng reviewer lý thuyết có thể bắt.

### 15. `\nocite{*}` làm Supplementary in toàn bộ 124 references

Dòng 2423:

```latex
\nocite{*}
```

khiến mọi entry trong `refs.bib` xuất hiện dù không được sử dụng. Dòng 90 đã có danh sách `\nocite{...}` cho references cần chia sẻ với main paper, nên `\nocite{*}` nhiều khả năng thừa và làm bibliography phình lớn.

### 16. Archive được khẳng định là “accompanying the submission”

Supplementary nhiều lần khẳng định có controlled-access verification archive đi kèm, đặc biệt dòng 100, 1923 và 2392. Trong main paper, câu tương ứng tại dòng 965 lại đang bị comment.

Nếu archive thực sự không được upload cho reviewer, đây là lỗi nghiêm trọng về claim/evidence boundary. Hoặc phải cung cấp archive/access instructions, hoặc sửa tất cả câu khẳng định thành kế hoạch release phù hợp với thực tế.

## V. Những phần tôi đã kiểm tra và chưa thấy lỗi nghiêm trọng

* Không có BibTeX key bị thiếu trong các lệnh `\cite`.
* Có 124 BibTeX entries và không có key trùng.
* Không có active label trùng trong main hoặc Supplementary.
* Các con số fallback (114+168+126+192=600), tỷ lệ 5.0%, các subtotal 40/60% đều nhất quán.
* Các Wilson intervals trong bảng collision khớp với counts.
* Title, abstract và contribution nhìn chung cùng một hướng nghiên cứu và đã phân biệt khá cẩn thận giữa replay, deployed policy và public-context comparisons.
* Việc tách open-loop nuScenes collision khỏi closed-loop CARLA collision được diễn đạt rõ.

## Thứ tự sửa ưu tiên

1. **Sửa budget gate bất khả thi và chạy lại các kết quả phụ thuộc.**
2. **Sửa Bradley–Terry calibration và (U_{\mathrm{pref}}^{\mathrm{pre}}).**
3. **Làm rõ/triển khai hợp lệ adaptive (N={4,6,8,10,12}).**
4. **Xóa kết luận 70.5% upper bound.**
5. **Đồng bộ flowchart với quality-only admission và giải quyết timing evidence.**
6. **Sửa source fatal/visible: section rỗng, input thiếu, biography image, Appendix M.**
7. **Sửa NAVSIM và SparseDrive trong bibliography.**

Ở trạng thái hiện tại, tôi **không khuyến nghị nộp ngay**, chủ yếu vì budget gate theo công thức hiện tại về mặt toán học sẽ fallback cho tất cả scene, trong khi toàn bộ kết quả deployed policy lại dựa trên việc gate này hoạt động bình thường.

[1]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/32768f7faf1995026ef9821c696f3404-Abstract-Datasets_and_Benchmarks_Track.html?utm_source=chatgpt.com "NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking"
[2]: https://arxiv.org/abs/2405.19620?utm_source=chatgpt.com "SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation"
