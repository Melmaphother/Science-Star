
class Constants:
    r"""A class containing constants used in CAMEL."""

    # This value defines the default size (both width and height) for images
    # extracted from a video.
    VIDEO_DEFAULT_IMAGE_SIZE = 768

    # This value defines the interval (in number of frames) at which images
    # are extracted from the video.
    VIDEO_IMAGE_EXTRACTION_INTERVAL = 50

    # Default plug of imageio to read video
    VIDEO_DEFAULT_PLUG_PYAV = "pyav"

    # Return response with json format
    FUNC_NAME_FOR_STRUCTURED_OUTPUT = "return_json_response"

    # Default top k vaule for RAG
    DEFAULT_TOP_K_RESULTS = 1

    # Default similarity threshold vaule for RAG
    DEFAULT_SIMILARITY_THRESHOLD = 0.7
