"""
Where the image goes in the Turn-1 prompt, shared by every evaluator.

    first (default)  <system> USER: <image>...<image>\\n<question> ASSISTANT:
    last  (old)      <system> USER: <question>\\n<image>...<image> ASSISTANT:

"first" is LLaVA's own evaluation layout.  With the image first, the image KV
does not depend on the question, and everything in front of the image is the
short system prompt.

An item may list more images in ``extra_images`` (make_long_context.py writes
them to make the context longer).  All images of an item form ONE contiguous
block of visual tokens: the item's own image first, then the extra ones.
"""

IMAGE_PLACEHOLDER = "<image>"
POSITIONS = ("first", "last")


def image_files(item):
    """File names of an item's images, its own image first."""
    own = item.get("image") or item.get("imagename")
    return [own] + list(item.get("extra_images") or [])


def first_turn_question(question, n_images=1, position="first"):
    """The Turn-1 user message with its image placeholders."""
    if position not in POSITIONS:
        raise ValueError(f"image position must be one of {POSITIONS}")
    text = (question or "").replace(IMAGE_PLACEHOLDER, "").strip()
    block = IMAGE_PLACEHOLDER * max(1, int(n_images))
    return block + "\n" + text if position == "first" else text + "\n" + block


def image_block(ids, image_token_index):
    """(start, count) of the contiguous run of image placeholders in a list of ids."""
    ids = [int(token) for token in ids]
    where = [index for index, token in enumerate(ids) if token == image_token_index]
    if not where:
        return None, 0
    if where[-1] - where[0] + 1 != len(where):
        raise ValueError(
            "The image placeholders must be next to each other (one visual block); "
            f"found them at positions {where}.")
    return where[0], len(where)


def add_arguments(parser):
    parser.add_argument(
        "--image-position", default="first", choices=list(POSITIONS),
        help="Turn-1 layout: image before the question (first, LLaVA's own layout) "
             "or after it (last, this repo's layout before November 2026).")
