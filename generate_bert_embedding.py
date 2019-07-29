from compare import pairDataToBertVecs

if __name__ == "__main__":
    pairDataToBertVecs(75, 10, "train")
    pairDataToBertVecs(20, 10, "test")