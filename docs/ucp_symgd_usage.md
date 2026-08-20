# Hướng dẫn chạy SemiSAM-O1 với UCP/SymGD

## Phạm vi

UCP/SymGD là nhánh huấn luyện tùy chọn dành cho `mt` và `uamt`. Tính năng mặc
định tắt và bắt đầu từ Round 2 khi bật. KNN refinement giữa các round vẫn hoạt
động ở cấp volume và không thay đổi.

## Chạy MT

Từ thư mục `code/`:

```powershell
python train_SemiSAM_O1.py `
  --root_path ../data/LA `
  --exp SemiSAM_O1_UCP/LA `
  --backbone mt `
  --max_iterations 15000 `
  --num_rounds 3 `
  --sam_ckpt pretrained_ckpt/sam_med3d_turbo.pth `
  --seed 1337 `
  --ucp_symgd
```

## Chạy UAMT

```powershell
python train_SemiSAM_O1.py `
  --root_path ../data/LA `
  --exp SemiSAM_O1_UCP/LA `
  --backbone uamt `
  --max_iterations 15000 `
  --num_rounds 3 `
  --sam_ckpt pretrained_ckpt/sam_med3d_turbo.pth `
  --seed 1337 `
  --ucp_symgd
```

## Tham số mới

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `--ucp_symgd` | tắt | Bật UCP và SymGD. |
| `--ucp_start_round` | `2` | Round đầu tiên dùng nhánh mới; nhánh hoạt động khi `round_num >=` giá trị này. |
| `--ucp_scale_min` | `0.3` | Tỷ lệ cạnh nhỏ nhất của khối UCP ở tâm. |
| `--ucp_scale_max` | `0.6` | Tỷ lệ cạnh lớn nhất của khối UCP ở tâm. |
| `--symgd_confidence` | `0.95` | Confidence tối thiểu trên cả hai góc nhìn teacher. |
| `--symgd_weight` | `1.0` | Trọng số SymGD tối đa ở cuối round; trọng số ramp từ 10% đến giá trị này. |

Giữ các giá trị mặc định cho lần chạy đầu tiên. Chỉ điều chỉnh sau khi đã có
kết quả baseline cùng seed và cùng cấu hình dữ liệu.

## So sánh baseline

Chạy lại cùng lệnh nhưng bỏ `--ucp_symgd`. Dùng `--exp` khác nhau để tránh ghi
đè checkpoint. So sánh Dice tốt nhất theo từng round và giữ nguyên seed,
template labeled, split dữ liệu cùng mọi tham số còn lại.

## Theo dõi

Trong các round được kích hoạt, TensorBoard ghi:

- `train/ucp_loss`;
- `train/symgd_loss`;
- `train/symgd_kept_ratio`;
- `train/symgd_weight`.

`symgd_kept_ratio` bằng 0 là hợp lệ khi teacher chưa có voxel vừa đồng thuận
vừa vượt ngưỡng confidence. Nếu tỷ lệ này luôn gần 0, thử giảm
`--symgd_confidence` sau khi hoàn tất baseline mặc định.

## Chi phí tài nguyên

Mỗi iteration ở round được kích hoạt thêm một forward student và một forward
EMA teacher trên batch gồm hai mixed volume. UAMT vốn đã chạy nhiều forward để
ước lượng uncertainty nên cần theo dõi VRAM chặt hơn MT. Khi thiếu VRAM, giảm
`--patch_size` theo bội số phù hợp với bốn lần downsampling của U-Net. Giữ
`--batch_size` ở mức tối thiểu 2 để vẫn có một phần labeled và một phần
unlabeled; nếu cần giảm thêm mức dùng bộ nhớ, ưu tiên giảm patch size hoặc các
tham số cấu hình khác.

## Kiểm tra code trên CPU

Từ thư mục gốc repository:

```powershell
python -m unittest discover -s code/tests -p "test_*.py" -v
python -m compileall -q code
```

## Smoke test trên máy CUDA

Smoke test sau đi qua cả Round 1 và nhánh UCP/SymGD ở Round 2:

```powershell
cd code
python train_SemiSAM_O1.py `
  --root_path ../data/LA `
  --exp Smoke_UCP/LA `
  --backbone mt `
  --max_iterations 2 `
  --num_rounds 2 `
  --sam_ckpt pretrained_ckpt/sam_med3d_turbo.pth `
  --ucp_symgd
```

Smoke test chỉ kiểm tra luồng chạy, shape tensor và backward; không dùng kết
quả Dice của lượt chạy ngắn này để đánh giá phương pháp.
