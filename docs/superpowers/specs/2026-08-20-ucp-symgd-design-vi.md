# Thiết kế tích hợp UCP/SymGD

## Mục tiêu

Bổ sung một nhánh huấn luyện Unified Copy-Paste (UCP) và Symmetric Guidance
(SymGD) dạng tùy chọn cho hai backbone MT và UAMT của SemiSAM-O1. Theo mặc
định, nhánh này bắt đầu từ Round 2, không làm thay đổi luồng huấn luyện và tinh
chỉnh KNN gốc khi bị tắt, đồng thời không gọi SAM trong quá trình huấn luyện
online.

## Phạm vi

Phiên bản đầu tiên hỗ trợ `mt` và `uamt`. DAN và DTC không nằm trong phạm vi vì
hai backbone này hiện không có mô hình EMA teacher; việc bổ sung teacher sẽ làm
thay đổi thuật toán và mức sử dụng bộ nhớ của chúng.

Thay đổi chỉ bao gồm UCP và SymGD trong lúc huấn luyện. Mặt nạ XOR theo voxel
không được đưa vào bước tinh chỉnh KNN. SemiSAM-O1 xếp hạng uncertainty theo
từng volume bằng entropy voxel trung bình và thay thế các volume không chắc
chắn thông qua KNN voting trên đặc trưng global, trong khi SymGD sử dụng vùng
bất đồng làm mặt nạ huấn luyện theo voxel.

## Giao diện dòng lệnh

Các tham số sau được thêm vào `code/train_SemiSAM_O1.py`:

- `--ucp_symgd`: cờ bật tính năng; mặc định tắt.
- `--ucp_start_round`: round đầu tiên kích hoạt nhánh; mặc định `2`.
- `--ucp_scale_min`: tỷ lệ chiều dài cạnh nhỏ nhất của khối hộp ở tâm; mặc định
  `0.3`.
- `--ucp_scale_max`: tỷ lệ chiều dài cạnh lớn nhất của khối hộp ở tâm; mặc định
  `0.6`.
- `--symgd_confidence`: confidence tối thiểu của teacher trên cả góc nhìn trực
  tiếp và góc nhìn ghép; mặc định `0.95`.
- `--symgd_weight`: trọng số symmetric-guidance tối đa; mặc định `1.0`.

Cấu hình được kiểm tra khi khởi động với các điều kiện:

- `ucp_start_round >= 1`;
- `0 < ucp_scale_min <= ucp_scale_max <= 1`;
- `0 <= symgd_confidence <= 1`;
- `symgd_weight >= 0`;
- chỉ được dùng `--ucp_symgd` với `mt` hoặc `uamt`.

Cấu hình không hợp lệ sẽ dừng trước khi tải dữ liệu hoặc mô hình. Khi không có
cờ bật tính năng, đường tính toán thông thường của MT, UAMT, DAN và DTC không
thay đổi về chức năng.

## Các thành phần

### Tiện ích UCP/SymGD thuần tensor

Tạo `code/utils/ucp_symgd.py` gồm các hàm tensor nhỏ có thể chạy trên CPU hoặc
CUDA:

1. Tạo một mặt nạ khối hộp 3D ở tâm cho mỗi mẫu unlabeled. Tỷ lệ chiều dài mỗi
   cạnh được lấy ngẫu nhiên từ phân phối đều trong khoảng cấu hình. Mặt nạ trả
   về có shape `[N_u, 1, D, H, W]`, cùng device và dtype với ảnh đầu vào.
2. Ghép mỗi mẫu unlabeled với một mẫu labeled bằng cách quay vòng chỉ số batch
   labeled. Cách này hỗ trợ batch mặc định `1 + 1` và tránh lỗi đặc biệt khi số
   lượng hai phần của batch thay đổi.
3. Sinh ảnh và nhãn inward/outward:

   ```text
   U_in  = X_l * M       + X_u * (1 - M)
   Q_in  = Y_l * M       + Y_u * (1 - M)
   U_out = X_u * M       + X_l * (1 - M)
   Q_out = Y_u * M       + Y_l * (1 - M)
   ```

4. Khôi phục góc nhìn unlabeled của teacher từ hai dự đoán đã trộn:

   ```text
   P_merged = P_out * M + P_in * (1 - M)
   ```

5. Tạo mặt nạ symmetric guidance. Một voxel chỉ được giữ khi nhãn argmax của
   teacher trên góc nhìn trực tiếp và góc nhìn ghép giống nhau, đồng thời xác
   suất softmax lớn nhất của cả hai đạt `symgd_confidence`.
6. Tính masked cross entropy từ logits trực tiếp của student đến hard label đã
   detach của teacher ghép. Nếu mặt nạ rỗng, trả về số không vẫn nối với đồ thị
   gradient thay vì thực hiện phép chia cho không.

Các tiện ích không chứa logic khởi tạo mô hình, tải dữ liệu, global state hoặc
dependency mới.

### Nhánh huấn luyện dùng chung

Thêm một helper dùng chung trong `code/train_SemiSAM_O1.py` cho MT và UAMT.
Helper nhận student, EMA teacher, batch hiện tại, các CE/Dice loss có sẵn, chỉ
số iteration và số round. Khi tính năng chưa hoạt động, helper trả về các loss
bằng không mà không chạy thêm forward pass.

Khi hoạt động, helper thực hiện:

1. Chia batch đã augmentation theo `labeled_bs`.
2. Tạo ảnh UCP inward/outward và hard target từ nhãn thật cùng pseudo-label đã
   tinh chỉnh của round hiện tại.
3. Chạy một student forward trên batch ghép chứa cả hai nhóm ảnh mixed.
4. Chạy một EMA teacher forward không gradient trên cùng batch mixed.
5. Tính trung bình CE+Dice loss của hai đầu ra inward và outward của student.
6. Khôi phục góc nhìn unlabeled ghép của teacher, sau đó tính SymGD loss đã lọc
   theo confidence và mức đồng thuận với đầu ra unlabeled trực tiếp hiện có của
   student.
7. Trả về UCP loss, SymGD loss và tỷ lệ voxel được giữ để ghi log.

Dự đoán trực tiếp của teacher đã được MT/UAMT tính sẵn sẽ được tái sử dụng.
Không bổ sung direct teacher forward nào khác.

## Tích hợp hàm mất mát

Hàm mất mát hiện có của từng backbone được giữ nguyên. Khi nhánh mới hoạt động:

```text
L_total = L_existing
        + pseudo_weight * L_ucp
        + gamma(t) * L_sym
```

`pseudo_weight` là linear ramp hiện có, tăng từ 0 lên 1 trong 30% iteration đầu
của mỗi round. Hệ số SymGD dùng lịch tăng tuyến tính đã được duyệt:

```text
gamma(t) = symgd_weight * (0.1 + 0.9 * clamp(t / max_iterations, 0, 1))
```

Xác suất teacher và hard label ghép đều được detach. Gradient chỉ đi qua dự
đoán trực tiếp của student đối với `L_sym` và hai dự đoán mixed của student đối
với `L_ucp`.

## Luồng dữ liệu và các bất biến

`TwoStreamBatchSampler` hiện có luôn đặt các mẫu labeled trước các mẫu
unlabeled. UCP chạy sau các phép xoay, lật, crop ngẫu nhiên và chuyển tensor
hiện có, nên mọi ảnh và nhãn được trộn có cùng shape không gian. Thay đổi này
không đụng đến dataset, sampler, dictionary pseudo-label, đặc trưng SAM,
validation, định dạng checkpoint, resume hoặc bước tinh chỉnh KNN giữa các
round.

Phần triển khai vẫn hỗ trợ multiclass: hard label sử dụng class ID dạng số
nguyên, quyết định của teacher dùng softmax/argmax, và loss dùng cross entropy
thay cho công thức BCE chỉ dành cho bài toán nhị phân.

## Ghi log

Trong các round được kích hoạt, ghi thêm những giá trị sau bên cạnh các loss
hiện có:

- UCP CE+Dice loss;
- SymGD masked cross-entropy loss;
- tỷ lệ voxel được mặt nạ SymGD giữ lại;
- trọng số SymGD hiện tại.

Tên scalar TensorBoard dùng tiền tố `train/`; text log bổ sung các giá trị mới
vào thông báo mỗi 100 iteration. Khi tính năng tắt hoặc đang ở Round 1, chương
trình không ghi metric khiến người dùng hiểu nhầm rằng nhánh mới đang hoạt
động.

## Xử lý lỗi

Khoảng giá trị CLI không hợp lệ hoặc backbone không được hỗ trợ sẽ gây
`ValueError` trước khi truy cập CUDA, checkpoint hoặc dữ liệu. Các tiện ích
runtime kiểm tra số chiều tensor, shape không gian tương thích, phần labeled và
unlabeled không rỗng, cùng shape class-logit. Thông báo lỗi nêu rõ bất biến bị
vi phạm.

Mặt nạ confidence/agreement rỗng là một trạng thái huấn luyện hợp lệ. Trong
trường hợp này, symmetric loss và tỷ lệ voxel được giữ đều bằng không.

## Xác minh

Tạo `code/tests/test_ucp_symgd.py` bằng test runner `unittest` thuộc thư viện
chuẩn của Python. Các test CPU bao phủ:

- hình học của khối hộp ở tâm và giới hạn scale;
- phép trộn ảnh và nhãn inward/outward;
- quay vòng mẫu labeled khi kích thước hai phần batch khác nhau;
- khôi phục chính xác góc nhìn unlabeled từ dự đoán teacher mixed;
- loại bỏ voxel bất đồng hoặc có confidence thấp;
- loss bằng không hữu hạn và vẫn hỗ trợ gradient khi mặt nạ rỗng;
- masked cross-entropy multiclass và luồng gradient.

Chạy:

```powershell
python -m unittest discover -s code/tests -p "test_*.py" -v
python -m compileall -q code
```

Máy hiện tại chỉ có PyTorch CPU và không có volume huấn luyện HDF5, vì vậy
không thể xác minh bằng một lượt train CUDA cục bộ. Tài liệu sử dụng sẽ cung
cấp lệnh smoke test MT/UAMT ngắn để người dùng chạy trong môi trường huấn
luyện đích.

## Tài liệu hướng dẫn

Tạo `docs/ucp_symgd_usage.md` bao gồm:

- lệnh baseline và lệnh bật tính năng cho MT/UAMT;
- toàn bộ tham số mới cùng giá trị mặc định;
- giải thích nhánh bắt đầu từ Round 2 trừ khi được cấu hình khác;
- số forward pass bổ sung và chi phí VRAM dự kiến;
- các metric TensorBoard cần theo dõi;
- lệnh xác minh CPU và quy trình smoke test CUDA ngắn;
- ghi chú rõ rằng KNN refinement vẫn hoạt động ở cấp sample và không thay đổi.

## Nguồn tham khảo

- Paper SemiSAM-O1: <https://arxiv.org/html/2604.24109>
- Paper SymGD: <https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Constructing_and_Exploring_Intermediate_Domains_in_Mixed_Domain_Semi-supervised_Medical_CVPR_2024_paper.html>
- Mã tham khảo SymGD: <https://github.com/MQinghe/MiDSS>
