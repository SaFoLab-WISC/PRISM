import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser(description="Make the image path to absolute path to support ms-swift.")
    parser.add_argument("--folder", help="Folder to the PRISM dataset.")
    args = parser.parse_args()

    data = json.load(open(os.path.join(args.folder, "data.json"), "r"))
    image_folder = os.path.join(args.folder, "images")
    swift_path = os.path.join(args.folder, "swift_data.json")

    for item in data:
        swift_data = {"messages": [{"role": "user", "content": f"<image>{item['question']}"}, {"role": "assistant", "content": item["response"]}], "images": [os.path.join(image_folder, item["image"])]}
        with open(swift_path, 'a') as f:
            json.dump(swift_data, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"Finish formatting data into ms-swift format into {swift_path}.")

if __name__ == "__main__":
    main()