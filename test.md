# Góp ý nội bộ trước khi nộp bài

## Nhận định chung

Sau khi rà soát lại `Submission.tex`, `Supplementary.tex` và `refs.bib`, tôi cho rằng bản hiện tại **chưa nên nộp ngay**. Một số lỗi chỉ thuộc về LaTeX hoặc đóng gói file, nhưng có vài điểm liên quan trực tiếp đến tính đúng đắn của phương pháp và khả năng tái lập kết quả.

Các vấn đề cần ưu tiên cao nhất là:

- budget gate hiện đang mâu thuẫn với chính các ngưỡng được khai báo;
- cách calibration của Bradley–Terry chưa khớp với đại lượng được dùng khi suy luận;
- biến \(U_{\mathrm{pref}}^{\mathrm{pre}}\) được dùng để chọn budget nhưng chưa được định nghĩa đầy đủ;
- mô hình được mô tả là student 6-step nhưng lại chạy với nhiều budget khác nhau;
- cách diễn giải \(0.86\times0.82=70.5\%\) là “upper bound” không đúng về xác suất;
- phần mô tả latency gate chưa thống nhất giữa bài chính, supplementary và flowchart.

Dưới đây là từng vấn đề cụ thể.

---

# 1. Budget gate đang tự mâu thuẫn và có thể khiến mọi scene đều fallback

## Chỗ đang mâu thuẫn

Trong Supplementary, đại lượng pre-run anchor distance được định nghĩa dưới dạng:

\[
\widehat d_{\mathrm{anch}}^{\mathrm{pre}}
=
\max \widehat r_{\mathrm{cell}}
+r_{\mathrm{noise}}
+\lambda_{\mathrm{env}}^\top u_{\mathrm{env}}.
\]

Trong khi đó:

- \(r_{\mathrm{noise}}=2.93\) m cho một thiết lập;
- \(r_{\mathrm{noise}}=4.18\) m cho thiết lập còn lại;
- scene chỉ được chấp nhận khi
  \[
  \widehat d_{\mathrm{anch}}^{\mathrm{pre}}\le R_{\mathrm{basin}};
  \]
- nhưng lại đặt
  \[
  R_{\mathrm{basin}}=2.50\text{ m}.
  \]

Theo đúng công thức trên:

\[
\widehat d_{\mathrm{anch}}^{\mathrm{pre}}\ge r_{\mathrm{noise}}.
\]

Do đó:

- trường hợp thứ nhất luôn có
  \[
  \widehat d_{\mathrm{anch}}^{\mathrm{pre}}\ge2.93>2.50;
  \]
- trường hợp thứ hai luôn có
  \[
  \widehat d_{\mathrm{anch}}^{\mathrm{pre}}\ge4.18>2.50.
  \]

Nếu implementation đúng như mô tả, gate này sẽ từ chối toàn bộ scene, tức fallback phải gần 100%. Điều này mâu thuẫn trực tiếp với fallback rate khoảng 5% được báo cáo.

## Vấn đề bổ sung về phân vị Gaussian

Supplementary ghi các phân vị:

\[
\chi_{12,0.95}=3.50,\qquad
\chi_{24,0.95}=4.99.
\]

Nếu đây thực sự là 95th percentile của Euclidean norm của Gaussian chuẩn \(D\)-chiều, các giá trị đúng xấp xỉ:

\[
\chi_{12,0.95}\approx4.585,
\qquad
\chi_{24,0.95}\approx6.034.
\]

Khi nhân với \(\sqrt{0.7}\), bán kính nhiễu phải xấp xỉ:

\[
r_{\mathrm{noise}}\approx3.84\text{ m}
\]

và

\[
r_{\mathrm{noise}}\approx5.05\text{ m}.
\]

Như vậy mâu thuẫn với \(R_{\mathrm{basin}}=2.50\) còn lớn hơn.

## Hướng giải quyết

Chúng ta cần kiểm tra lại đúng đại lượng đã dùng trong code:

- Euclidean norm của toàn bộ vector trajectory;
- RMS theo số chiều;
- per-step displacement;
- hay một normalized trajectory distance khác.

Có khả năng công thức dự kiến là RMS:

\[
\frac{\|\epsilon\|_2}{\sqrt D},
\]

nhưng phần \(\sqrt D\) đã bị thiếu trong manuscript.

Sau khi xác định đúng đại lượng, cần làm lại theo thứ tự:

- sửa định nghĩa toán học;
- tính lại \(r_{\mathrm{noise}}\);
- hiệu chỉnh lại \(R_{\mathrm{basin}}\);
- chạy lại scene admission;
- cập nhật phân bố budget và fallback rate;
- kiểm tra lại toàn bộ bảng CARLA hoặc các kết quả phụ thuộc vào adaptive budget.

Không nên chỉ sửa \(R_{\mathrm{basin}}\) trên giấy nếu kết quả hiện tại được sinh từ một công thức khác.

---

# 2. Calibration của Bradley–Terry chưa khớp với probability dùng khi suy luận

## Chỗ đang mâu thuẫn

Bradley–Terry được huấn luyện từ pairwise margin:

\[
s_{ij}
=
\frac{
r_\phi(o,\tau_i)-r_\phi(o,\tau_j)
}{
\beta_{\mathrm{BT}}
}.
\]

Loss này chỉ học sự khác biệt giữa hai utility. Với một hàm dịch phụ thuộc observation \(c(o)\):

\[
r'_\phi(o,\tau)
=
r_\phi(o,\tau)+c(o),
\]

thì mọi pairwise margin vẫn giữ nguyên. Vì vậy giá trị tuyệt đối \(r_\phi(o,\tau)\) không được xác định duy nhất.

Tuy nhiên ở runtime, bài lại dùng:

\[
p_k^{\mathrm{cal}}
=
C_{\mathrm{iso}}
\left(
r_\phi(o,\tau_k)
\right),
\]

sau đó chuyển probability này thành log-odds để chấm điểm candidate.

Điểm không thống nhất ở đây là:

- training và calibration được mô tả trên pairwise comparison;
- ECE và Brier score cũng được báo cáo trên các pair;
- nhưng runtime lại sử dụng probability tuyệt đối của từng candidate.

Như vậy ECE và Brier hiện tại chưa chứng minh rằng probability dùng trong deployed selector đã được calibrated đúng.

## Hướng giải quyết

Có ba phương án hợp lệ.

### Phương án 1: Giữ Bradley–Terry theo pairwise form

Dùng trực tiếp:

\[
P(\tau_i\succ\tau_j\mid o)
=
\sigma
\left(
\frac{r_i-r_j}{\beta}
\right),
\]

sau đó chọn candidate bằng pairwise tournament, Copeland score hoặc aggregate pairwise win probability.

Đây là phương án gần nhất với formulation hiện tại.

### Phương án 2: Dùng một reference trajectory cố định

Định nghĩa:

\[
s_k
=
r(o,\tau_k)-r(o,\tau_{\mathrm{ref}}),
\]

rồi calibration \(s_k\) thay vì raw utility.

Reference trajectory phải được định nghĩa rõ và nhất quán ở cả training, calibration và runtime.

### Phương án 3: Center utility theo candidate set

Ví dụ:

\[
\bar r_k
=
r_k-\frac{1}{K}\sum_j r_j.
\]

Sau đó isotonic calibration phải được fit trực tiếp trên \(\bar r_k\), với nhãn candidate-level phù hợp.

## Việc cần cập nhật sau khi sửa

- định nghĩa chính xác input của isotonic calibration;
- cập nhật ECE, Brier score và NLL;
- kiểm tra lại các threshold preference;
- chạy lại selector ablation;
- cập nhật CARLA results nếu ranking hoặc fallback thay đổi.

---

# 3. \(U_{\mathrm{pref}}^{\mathrm{pre}}\) được dùng nhưng chưa được định nghĩa

## Chỗ đang thiếu

Đại lượng

\[
U_{\mathrm{pref}}^{\mathrm{pre}}
\]

được dùng trong pre-run scene gate và có threshold:

\[
U_{\mathrm{pref}}^{\mathrm{pre}}\le0.28.
\]

Tuy nhiên manuscript chưa cho biết:

- input của uncertainty head là gì;
- uncertainty này được tính trước khi sinh candidate như thế nào;
- target label là gì;
- loss function là gì;
- calibration được thực hiện thế nào;
- metric đánh giá là ECE, Brier, AUROC hay coverage;
- quan hệ giữa \(U_{\mathrm{pref}}^{\mathrm{pre}}\) và post-candidate uncertainty là gì.

Đây không phải biến phụ. Nó tham gia trực tiếp vào quyết định chọn budget, nên nếu thiếu định nghĩa, adaptive policy không thể được tái lập.

## Mâu thuẫn bổ sung

Bảng threshold có một hard gate riêng:

\[
u_{\mathrm{anch}}\le0.40,
\]

nhưng phương trình scene admission lại không chứa điều kiện này dưới dạng hard constraint. \(u_{\mathrm{anch}}\) chỉ xuất hiện trong một weighted sum của anchor-distance predictor.

Hai cơ chế này không tương đương.

## Hướng giải quyết

Cần bổ sung một subsection ngắn gồm:

- định nghĩa chính xác \(U_{\mathrm{pref}}^{\mathrm{pre}}\);
- kiến trúc hoặc ít nhất input/output của head;
- training target;
- loss;
- calibration protocol;
- threshold-selection protocol;
- performance trên validation/test;
- cách nó được dùng trong Algorithm.

Đồng thời phải thống nhất một trong hai cách đối với \(u_{\mathrm{anch}}\):

- dùng hard gate rõ ràng trong scene admission equation; hoặc
- bỏ threshold hard gate khỏi bảng và chỉ giữ weighted contribution.

---

# 4. “6-step student” không thống nhất với adaptive budget \(\{4,6,8,10,12\}\)

## Chỗ đang mâu thuẫn

Supplementary mô tả progressive consistency distillation:

\[
50\rightarrow25\rightarrow12\rightarrow6,
\]

và gọi model cuối là “6-step student”.

Tuy nhiên deployed policy lại chọn:

\[
N\in\{4,6,8,10,12\}.
\]

Algorithm sau đó dường như sử dụng cùng một student cho tất cả giá trị \(N\).

Điểm chưa rõ là:

- có một checkpoint riêng cho từng \(N\) hay không;
- checkpoint 6-step có được phép chạy 4, 8, 10 hoặc 12 bước hay không;
- timestep schedule cho từng \(N\) được xây dựng thế nào;
- các budget 4, 8 và 10 có xuất hiện trong training không;
- predictor \(g_{\psi,i,N}\) được fit trên output của model nào.

Nếu chỉ có một fixed-step distilled student, không thể mặc nhiên suy ra rằng model đó hoạt động hợp lệ với arbitrary step count.

## Hướng giải quyết

Cần xác định đúng implementation thuộc trường hợp nào.

### Trường hợp A: Có nhiều checkpoint

Ghi rõ:

- checkpoint cho \(N=4\);
- checkpoint cho \(N=6\);
- checkpoint cho \(N=8\);
- checkpoint cho \(N=10\);
- checkpoint cho \(N=12\).

Đồng thời bổ sung chi phí lưu trữ và cách runtime load/chọn checkpoint.

### Trường hợp B: Có một variable-step model

Cần mô tả:

- training có conditioning theo \(N\) hay timestep grid hay không;
- sampling schedule cho từng \(N\);
- loss đảm bảo cross-budget consistency;
- validation riêng cho từng budget.

### Trường hợp C: Chỉ có student 6-step

Nếu implementation thực tế chỉ hỗ trợ 6 bước, cần bỏ claim adaptive \(\{4,6,8,10,12\}\), hoặc huấn luyện bổ sung trước khi giữ kết quả này.

---

# 5. \(0.86\times0.82=70.5\%\) không phải “upper bound”

## Chỗ đang sai

Bài hiện diễn giải:

\[
0.86\times0.82\approx70.5\%
\]

là “BT-vs-majority-human upper bound”.

Cách diễn giải này không đúng.

Giả sử:

\[
P(\mathrm{BT}=\mathrm{VLM})=0.86,
\]

và

\[
P(\mathrm{VLM}=\mathrm{Human})=0.82.
\]

Tích \(0.86\times0.82\) chỉ tương ứng với một giả thiết độc lập rất mạnh và chỉ tính trường hợp cả BT và human cùng đồng ý với VLM. Nó bỏ qua trường hợp BT và human cùng bất đồng với VLM nhưng lại đồng ý với nhau.

Do đó tích này:

- không phải upper bound;
- không phải lower bound chuẩn;
- không phải BT–human agreement trừ khi đưa thêm các giả thiết rất cụ thể.

## Hướng giải quyết

Ưu tiên tốt nhất là đánh giá trực tiếp:

\[
P(\mathrm{BT}=\mathrm{majority\ human})
\]

trên đúng 500 pair đã có human annotation.

Nếu chưa có phép đo trực tiếp, nên:

- xóa hàng “upper bound”;
- không dùng 70.5% như một accuracy estimate;
- chỉ ghi trung tính rằng đây là tích của hai agreement rates, nếu thật sự cần giữ;
- tránh diễn giải nó như evidence về human alignment.

---

# 6. Latency gate chưa thống nhất giữa bài chính, supplementary và flowchart

## Chỗ đang mâu thuẫn

Một đoạn trong bài chính nói deployed headline policy chỉ sử dụng quality admission và không dùng latency gate.

Tuy nhiên caption của flowchart lại nói latency gate và quality gate cùng xác định admissible budget set.

Supplementary cũng cho biết:

- \(N\ge8\) vượt 28 ms;
- latency tại \(N=12\) chỉ được linear projection;
- p50 và p99 của adaptive deployment chưa được đo đầy đủ;
- measurement được để lại cho camera-ready.

Như vậy hiện tại có ba cách diễn giải khác nhau:

- deployed policy không dùng latency;
- flowchart nói có dùng latency;
- timing evidence cho budget lớn lại chưa được đo thực tế.

## Hướng giải quyết

Cần chọn một narrative duy nhất.

### Nếu headline policy thực sự chỉ quality-based

- bỏ latency gate khỏi flowchart;
- sửa caption;
- ghi latency analysis là diagnostic/offline analysis;
- không gọi budget set là jointly constrained by latency.

### Nếu policy thực sự dùng latency gate

- bổ sung latency constraint vào phương trình admissible set;
- đo latency thực tế cho từng \(N\);
- báo p50, p95 hoặc p99;
- không dùng linear projection thay cho measurement đối với \(N=12\);
- cập nhật budget distribution sau khi thêm latency gate.

Câu “left to camera-ready measurement” nên bỏ. Những kết quả quan trọng phải có trong bản đang được reviewer đánh giá.

---

# 7. Có một `Introduction` rỗng ngay trước tài liệu tham khảo

## Chỗ đang sai

Trong `Submission.tex` có thêm:

```latex
\section{Introduction}
```

ngay sau Conclusion và trước References.

Đây rõ ràng là section thừa.

## Ảnh hưởng

- làm sai cấu trúc section;
- có thể tạo một mục rỗng trong PDF;
- khiến reviewer nghi ngờ file source chưa được kiểm tra lần cuối.

## Hướng giải quyết

Xóa hoàn toàn dòng section này.

---

# 8. Ảnh biography của tác giả đầu tiên đang trỏ nhầm sang figure tổng quan

## Chỗ đang sai

Biography của Tran Duc Anh đang sử dụng:

```latex
{image/fig1_vinad_overview_drawn.pdf}
```

Đây là hình tổng quan phương pháp, không phải ảnh chân dung.

## Hướng giải quyết

- thay bằng ảnh tác giả đúng;
- hoặc bỏ image option khỏi biography nếu chưa có ảnh.

---

# 9. Submission archive hiện chưa tự biên dịch đầy đủ

## Các file đang thiếu hoặc chưa nhất quán

Main file có:

```latex
\input{tables/per_n_latency_projection}
```

nhưng file tương ứng không có trong bộ đã gửi.

Ngoài ra còn thiếu:

- các graphics của main paper;
- graphics trong supplementary;
- ảnh biography;
- class file nếu không dùng template package chuẩn có sẵn.

Supplementary có thể compile ở draft-graphics mode, nhưng xuất hiện:

- duplicate hyperlink destinations trong algorithm line numbers;
- một số overfull boxes;
- bibliography rất dài do `\nocite{*}`.

## Hướng giải quyết

Trước khi upload, nên tạo một thư mục submission sạch và chạy compile từ đầu bằng:

```bash
latexmk -pdf Submission.tex
latexmk -pdf Supplementary.tex
```

Sau đó kiểm tra:

- không missing file;
- không undefined reference;
- không undefined citation;
- không duplicate destination;
- không overfull box đáng kể;
- tất cả hình và bảng input đều có trong archive.

---

# 10. Hard-coded “Appendix M” đang trỏ sai

## Chỗ đang mâu thuẫn

Main paper ghi rằng một diagnostic được trình bày trong “Appendix M”.

Tuy nhiên theo thứ tự appendix hiện tại, nội dung đó nằm ở appendix khác. Appendix M đang chứa phần Step Pareto and Anchor Evidence.

## Hướng giải quyết

Không nên ghi cứng ký tự appendix.

Thay bằng:

```latex
Appendix~\ref{sec:app_theory_proofs}
```

hoặc label tương ứng với đúng subsection.

---

# 11. Một số entry trong `refs.bib` có metadata sai

## NAVSIM

Entry NAVSIM hiện có danh sách tác giả không đúng với paper chính thức.

Cần kiểm tra lại:

- toàn bộ author list;
- venue;
- year;
- title capitalization;
- publication status.

## SparseDrive

Entry SparseDrive có ít nhất hai lỗi:

- tên tác giả đầu đang là “Wenyuan Sun”, trong khi paper dùng “Wenchao Sun”;
- title không đúng hoàn toàn với title chính thức.

## Hướng giải quyết

Nên đối chiếu từng entry quan trọng với:

- publisher page;
- conference proceedings;
- arXiv bản chính thức;
- DOI metadata.

Không nên chỉ sửa hai entry này rồi dừng lại. Nên chạy thêm một lượt kiểm tra toàn bộ bibliography đối với các paper gần đây.

---

# 12. “Conformal slack” chưa đủ thông tin để tạo ra conformal guarantee

## Chỗ đang thiếu

Supplementary mô tả việc lấy empirical \(1-\delta\) quantile của route maxima, nhưng chưa nêu:

- kích thước calibration set;
- đơn vị exchangeability là frame, route hay scene;
- finite-sample quantile correction;
- target coverage;
- observed empirical coverage;
- cách xử lý route clustering.

Nếu không có các thông tin này, reviewer có thể cho rằng đây chỉ là empirical quantile calibration chứ chưa phải conformal prediction theo nghĩa có coverage guarantee.

## Hướng giải quyết

Có hai lựa chọn.

### Giữ từ “conformal”

Bổ sung đầy đủ:

- calibration split;
- nonconformity score;
- finite-sample order statistic;
- coverage theorem hoặc proposition;
- empirical coverage và confidence interval.

### Không muốn mở rộng lý thuyết

Đổi tên thành:

- route-cluster residual quantile slack;
- calibrated route-level residual margin;
- empirical rejection margin.

Cách này an toàn hơn nếu mục tiêu chỉ là engineering calibration.

---

# 13. Volumetric covering bound có thể đang viết sai chiều bất đẳng thức

## Chỗ cần kiểm tra

Bài viết rằng một volumetric covering bound “requires at least”

\[
(3/\eta)^D
\]

probes.

Tuy nhiên các công thức covering-number phổ biến thường được dùng để đưa ra upper bound cho kích thước của một \(\eta\)-net hoặc mô tả exponential scaling. Không thể mặc nhiên chuyển thành lower bound “requires at least” nếu không chỉ rõ theorem và assumptions.

## Hướng giải quyết

Cách viết an toàn hơn:

> The covering number grows exponentially with the ambient dimension \(D\).

Nếu cần giữ công thức, phải:

- trích đúng theorem;
- nêu norm;
- nêu miền được cover;
- chỉ rõ đây là upper bound hay lower bound.

---

# 14. `\nocite{*}` làm supplementary in toàn bộ bibliography

## Chỗ đang chưa hợp lý

Supplementary đã có một danh sách `\nocite{...}` để đưa các reference dùng chung với main paper vào bibliography.

Tuy nhiên cuối file lại có:

```latex
\nocite{*}
```

Lệnh này đưa toàn bộ entry trong `refs.bib` vào tài liệu, kể cả các tài liệu không được dùng.

## Ảnh hưởng

- bibliography bị phình lớn;
- khó kiểm tra tài liệu nào thực sự được sử dụng;
- có thể tạo cảm giác citation padding;
- làm tăng số trang supplementary không cần thiết.

## Hướng giải quyết

Xóa `\nocite{*}` và chỉ giữ:

- các citation thực sự xuất hiện;
- hoặc danh sách `\nocite{key1,key2,...}` có chủ đích.

---

# 15. Claim về verification archive phải khớp với tài liệu thực sự được nộp

## Chỗ cần kiểm tra

Supplementary nhiều lần nói verification archive “accompanies the submission”.

Trong main paper, câu tương ứng về archive lại đang bị comment.

Nếu archive không thực sự được upload cùng submission, đây sẽ là một claim không đúng.

## Hướng giải quyết

### Nếu archive có nộp

- bổ sung tên file;
- mô tả cấu trúc;
- hướng dẫn truy cập;
- xác nhận reviewer có quyền mở;
- đồng bộ statement giữa main và supplementary.

### Nếu archive chưa nộp

Thay các câu khẳng định bằng wording trung tính, ví dụ:

> A verification archive will be released upon publication, subject to the applicable access constraints.

Không nên viết “accompanies the submission” nếu reviewer không nhận được file đó.

---

# 16. Các điểm đã kiểm tra và chưa thấy mâu thuẫn nghiêm trọng

- Không thấy citation key bị thiếu trong các lệnh `\cite`.
- Không thấy BibTeX key trùng nhau.
- Không thấy active label trùng trong main hoặc supplementary.
- Tổng số fallback case trong bảng cộng đúng thành 600.
- Fallback percentage và các subtotal nhìn chung nhất quán.
- Wilson confidence intervals trong bảng collision khớp với counts.
- Bài đã phân biệt tương đối rõ:
  - open-loop replay;
  - closed-loop CARLA;
  - deployed policy;
  - diagnostic analysis;
  - public-context comparison.
- Title, abstract và contribution nhìn chung cùng một hướng nghiên cứu.

---

# Thứ tự sửa đề xuất

## Mức 1 — Phải sửa trước khi nộp

- Sửa budget gate và chạy lại toàn bộ kết quả phụ thuộc.
- Sửa Bradley–Terry calibration.
- Định nghĩa đầy đủ \(U_{\mathrm{pref}}^{\mathrm{pre}}\).
- Làm rõ cách model hỗ trợ \(N\in\{4,6,8,10,12\}\).
- Xóa diễn giải “70.5% upper bound”.
- Thống nhất latency gate giữa text, equations và flowchart.

## Mức 2 — Phải sửa trong source

- Xóa `Introduction` rỗng.
- Sửa ảnh biography.
- Sửa hard-coded Appendix M.
- Bổ sung file input và toàn bộ graphics.
- Xóa hoặc thay `\nocite{*}`.
- Kiểm tra archive statement.

## Mức 3 — Nên sửa để tránh reviewer bắt lỗi

- Chuẩn hóa lại NAVSIM và SparseDrive references.
- Làm rõ conformal terminology.
- Sửa cách phát biểu volumetric covering bound.
- Dọn duplicate hyperlink destinations và overfull boxes.
- Compile lại toàn bộ submission từ một thư mục sạch.

---

# Kết luận nội bộ

Điểm đáng lo nhất hiện nay là budget gate. Theo đúng phương trình và threshold đang viết, gate không thể chấp nhận bất kỳ scene nào, trong khi kết quả lại báo fallback khoảng 5%. Đây là mâu thuẫn trực tiếp giữa formulation và reported result, nên reviewer có thể đặt câu hỏi về toàn bộ adaptive deployment pipeline.

Sau budget gate, hai điểm cần xử lý ngay là Bradley–Terry calibration và cơ chế variable-budget student. Nếu ba phần này được sửa đầy đủ và kết quả được chạy lại nhất quán, phần còn lại chủ yếu là chỉnh source, diễn đạt và đóng gói submission.
