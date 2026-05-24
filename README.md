# Vietnamese Extractive News Summarization

Project này tóm tắt tin tức tiếng Việt bằng **Position-Aware LexRank + MMR**.

Phiên bản hiện tại hỗ trợ **multi-document summarization**: mỗi cluster là một sự kiện/bài báo được thu thập từ nhiều nguồn khác nhau.

Cấu trúc dữ liệu được hỗ trợ:

```text
data/
  original/
    Cluster_111/
      original/
        710.txt
        ...
  summary/
    Cluster_111/
      0.gold.txt
      1.gold.txt
```

- LexRank: biểu diễn mỗi câu bằng TF-IDF, tạo đồ thị cosine similarity giữa các câu, rồi chạy PageRank để tìm câu trung tâm.
- Position-aware: vector teleport/prior của PageRank ưu tiên câu xuất hiện sớm trong bài và có liên quan đến tiêu đề.
- MMR: chọn câu theo công thức cân bằng relevance và novelty để tránh chọn nhiều câu quá giống nhau.

## Chạy nhanh

```powershell
python position_aware_lexrank_mmr.py --input data --output outputs --max-sentences 5
```

Kết quả sẽ được ghi vào `outputs/Cluster_xxx_summary.txt`.

## Chạy kèm đánh giá ROUGE

Khi đánh giá cluster, script dùng các file `data/summary/Cluster_xxx/*.gold.txt` làm reference summaries:

```powershell
python position_aware_lexrank_mmr.py --input data --output outputs --max-sentences 5 --evaluate
```

Nếu không có thư mục `summary`, script sẽ fallback về trường `Summary:` trong các bài nguồn.

## Chạy bảng thí nghiệm cho báo cáo

```powershell
python position_aware_lexrank_mmr.py --input data --compare-methods --max-sentences 5
```

Xuất bảng ra CSV:

```powershell
python position_aware_lexrank_mmr.py --input data --compare-methods --max-sentences 5 --csv outputs/results.csv --save-summaries
```

Thêm `--save-summaries` nếu muốn vừa chạy bảng so sánh vừa ghi các file
`outputs/Cluster_xxx_summary.txt` cho phương pháp chính.

Lệnh này so sánh 4 phương pháp:

- `Lead-k`: baseline lấy `k` câu đầu.
- `Vanilla LexRank`: LexRank gốc với prior đều.
- `Position-Aware LexRank`: LexRank có prior theo vị trí và tiêu đề.
- `Position-Aware LexRank + MMR`: chọn câu bằng MMR để giảm trùng lặp.

Output gồm:

- ROUGE: mức giống với gold summary.
- `Redundancy`: độ giống cosine trung bình giữa các câu được chọn. Càng thấp thì càng ít lặp ý.
- `SrcCover`: tỉ lệ nguồn trong cluster có câu được chọn. Càng cao thì summary lấy thông tin từ nhiều nguồn hơn.

Với multi-document cluster, MMR đặc biệt phù hợp vì nhiều nguồn thường viết lại cùng một sự kiện bằng các câu rất giống nhau.

## Notebook

Mở `experiments.ipynb` để chạy toàn bộ thí nghiệm, bảng ablation, grid search tham số MMR và xem summary mẫu cho từng phương pháp.

## Tham số quan trọng

- `--method`: chọn một phương pháp: `lead`, `lexrank`, `position_lexrank`, `position_lexrank_mmr`.
- `--max-sentences`: số câu tối đa trong bản tóm tắt.
- `--threshold`: ngưỡng cosine để tạo cạnh trong đồ thị LexRank. Mặc định `0.1`.
- `--position-weight`: mức ưu tiên vị trí câu. Mặc định `0.8`; phần còn lại dùng độ giống tiêu đề.
- `--lambda-mmr`: cân bằng giữa relevance và diversity. Mặc định `0.7`.
  - Gần `1.0`: ưu tiên câu có điểm LexRank cao.
  - Gần `0.0`: phạt trùng lặp mạnh hơn.
- `--ratio`: dùng tỉ lệ câu thay vì số câu cố định, ví dụ `--ratio 0.25`.
- `--keep-list-sentences`: giữ lại các câu dạng danh sách/bảng. Mặc định script lọc các dòng liệt kê dài như đội hình cầu thủ.

## Công thức MMR

Với tập câu đã chọn là `S`, câu ứng viên `c` được chấm:

```text
MMR(c) = lambda * LexRank(c) - (1 - lambda) * max_sim(c, s), s in S
```

Câu đầu tiên thường là câu có relevance cao nhất. Các câu sau bị trừ điểm nếu quá giống những câu đã chọn.

Trong code, điểm LexRank được scale về `[0, 1]` trước khi đưa vào MMR để cùng thang với cosine similarity.
