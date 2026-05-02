from common import OptimalControlModel, expect_value_error


def test_cpp_identifier_validation_rejects_keywords_and_duplicates():
    def keyword_state():
        model = OptimalControlModel("KeywordModel")
        model.state("class")

    def duplicate_names():
        model = OptimalControlModel("DuplicateModel")
        model.state("x")
        model.control("x")

    def generated_temp_collision():
        model = OptimalControlModel("TempCollisionModel")
        model.state("dt")

    expect_value_error(keyword_state, "C++")
    expect_value_error(duplicate_names, "duplicate")
    expect_value_error(generated_temp_collision, "reserved")


if __name__ == "__main__":
    test_cpp_identifier_validation_rejects_keywords_and_duplicates()
