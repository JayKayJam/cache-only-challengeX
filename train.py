from rfdetr import RFDETRSmall

model = RFDETRSmall()


def main():
    model.train(
        dataset_dir=".",
        epochs=10,
        batch_size=4,
        grad_accum_steps=4,
        # device="cuda",        # CUDA may not be available, uncomment if so
        # grad_accum_steps=4    # may be useful later for different hardware
        lr=1e-4,
        output_dir="runs/drone_v1",
    )

if __name__ == "__main__":
    main()