import base64, collections, pdf2image, re, requests


# KEY = re.search(
#     'api_key:{[^:]+:"([^"]+)"}',
#     requests.get("https://explorer.apis.google.com/embedded.js").text
# )[1]
KEY = "AIzaSyAa8yy0GdcGPHdtD083HiGGx_S0vMPScDM"
ORIGIN = "https://explorer.apis.google.com"
ENDPOINT = "https://vision.googleapis.com/v1/files:annotate"
IMAGE_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"


def detect_pdf_text(content, pages):
    inputConfig = {
        "content": base64.b64encode(content).decode(),
        "mimeType": "application/pdf",
    }
    features = [{"type": "DOCUMENT_TEXT_DETECTION"}]

    responses = []

    with requests.Session() as session:
        session.headers = {"origin": ORIGIN}
        session.params = {"key": KEY}

        for i in range(1, pages + 1, 5):
            body = {
                "requests": [
                    {
                        "inputConfig": inputConfig,
                        "features": features,
                        "pages": list(range(i, min(i + 5, pages + 1))),
                    }
                ]
            }

            response = session.post(ENDPOINT, json=body)
            responses.extend(response.json()["responses"][0]["responses"])

    return [response["fullTextAnnotation"] for response in responses]


def get_top_left(bounding_box, width, height):
    vertices = bounding_box.get("normalizedVertices")
    if vertices:
        return (
            min(vertex["x"] for vertex in vertices),
            min(vertex["y"] for vertex in vertices),
        )

    vertices = bounding_box["vertices"]
    return (
        min(vertex["x"] for vertex in vertices) / width,
        min(vertex["y"] for vertex in vertices) / height,
    )


def seperate_lines(page):
    width, height = page["width"], page["height"]

    for block in page["blocks"]:
        for paragraph in block["paragraphs"]:
            line, top = [], float("inf")
            prev_left = 0

            for word in paragraph["words"]:
                top_left = get_top_left(word["boundingBox"], width, height)

                if top_left[0] < prev_left:
                    yield line, top
                    line, top = [], float("inf")

                prev_left = top_left[0]
                top = min(top, top_left[1])

                text = "".join(symbol["text"] for symbol in word["symbols"])
                line.append(text)

            if line:
                yield line, top


def find_questions(lines, *, pattern=r"([0-9]+)\."):
    for line, top in lines:
        match = re.fullmatch(pattern, line[0])

        if not match and len(line) > 1:
            match = re.match(pattern, line[0] + line[1])

        if match:
            yield int(match[1]), top


def get_questions(page):
    lines = seperate_lines(page["pages"][0])
    questions = find_questions(lines)
    return sorted(questions, key=lambda question: question[1])


def filter_questions(questions):
    last = 0
    section = 1

    for image, questions in enumerate(questions):
        questions.append((None, None))

        for (num, top), (next_num, _) in zip(questions, questions[1:]):
            if num == last + 1:
                pass
            elif next_num == num + 1:
                section += 1
            else:
                continue

            last = num
            yield "{}-{}".format(section, num), image, top


def slice_image(image, slices):
    for top, bottom in zip(slices, slices[1:]):
        yield image.crop((0, top, image.width, bottom))


def detect_images_text(paths):
    features = [{"type": "DOCUMENT_TEXT_DETECTION"}]

    responses = []

    with requests.Session() as session:
        session.headers = {"origin": ORIGIN}
        session.params = {"key": KEY}

        for i in range(0, len(paths), 16):
            body = {"requests": []}

            for path in paths[i : i + 16]:
                with open(path, "rb") as file:
                    content = file.read()

                body["requests"].append(
                    {
                        "image": {"content": base64.b64encode(content).decode()},
                        "features": features,
                    }
                )

            response = session.post(IMAGE_ENDPOINT, json=body)
            responses.extend(response.json()["responses"])

    return [response["fullTextAnnotation"]["text"] for response in responses]


def slice_pdf(path):
    images = pdf2image.convert_from_path(path)

    with open(path, "rb") as file:
        content = file.read()

    pages = detect_pdf_text(content, len(images))
    questions = (get_questions(page) for page in pages)

    slices = collections.defaultdict(list)

    for num, image, top in filter_questions(questions):
        slices[image].append([top * images[image].height, num])

    paths = []

    for image, slices in slices.items():
        slices = sorted(slices)
        tops, nums = [top for top, num in slices], (num for top, num in slices)
        tops.append(images[image].height)

        for num, slice in zip(nums, slice_image(images[image], tops)):
            image_path = "{}.{}.png".format(path, num)
            paths.append(image_path)
            slice.save(image_path)

    for path, text in zip(paths, detect_images_text(paths)):
        with open("{}.txt".format(path[:-4]), "w") as file:
            print(text, file=file)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Splits PDF files containing multiple-choice exams into their individual questions"
    )
    parser.add_argument("file", nargs="+", help="The PDF file(s) to be processed")

    args = parser.parse_args()

    for file in args.file:
        print("slicing", file)
        slice_pdf(file)
