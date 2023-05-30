from typing import Optional, Type

from langchain import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.graphs.networkx_graph import KG_TRIPLE_DELIMITER, NetworkxEntityGraph, parse_triples
from langchain.llms import BaseLLM, OpenAI
from langchain.prompts.prompt import PromptTemplate
from pydantic import BaseModel


_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE = (
    "You are a networked intelligence helping a human track knowledge triples"
    " about all relevant people, things, concepts, etc. and integrating"
    " them with your knowledge stored within your weights"
    " as well as that stored in a knowledge graph."
    " Extract all of the knowledge triples from the text."
    " A knowledge triple is a clause that contains a subject, a predicate,"
    " and an object). The subject is the entity being described,"
    " the predicate is the property of the subject that is being"
    " described, and the object is the value of the property.\n\n"
    "EXAMPLE\n"
    'Phong trào "tị địa" diễn ra rất sôi nổi, khiến cho Pháp gặp rất nhiều khó khăn trong việc tổ chức, quản lí những vùng đất chúng mới chiếm được. Các đội nghĩa quân vẫn không chịu hạ vũ khí mà hoạt động ngày càng mạnh mẽ. Cuộc khởi nghĩa Trương Định tiếp tục giành được thắng lợi, gây cho Pháp nhiều khó khăn. Trương Định là con của Lãnh binh Trương Cầm, quê ở Quảng Ngãi. Ông theo cha vào Nam từ hồi nhỏ. Năm 1850, ông cùng Nguyễn Tri Phương mộ phu đồn điền, khai khẩn nhiều đất đai, được triều đinh phong chức Phó Quản cơ. Năm 1859, khi Pháp đánh Gia Định, Trương Định đã đưa đội quân đồn điền của ông về sát cánh cùng quân triều đình chiến đấu. Tháng 3 - 1860, khi Nguyễn Tri Phương được điều vào Gia Định, ông lại chủ động đem quân phối hợp đánh địch. Tháng 2 - 1861, chiến tuyến Chí Hoà bị vỡ, ông đưa quân về hoạt động ở Tân Hoà (Gò Công), quyết tâm chiến đấu lâu dài. Sau Hiệp ước 1862, triều đình hạ lệnh cho Trương Định phải bãi binh, mặt khác điều ông đi nhận chức Lãnh binh ở An Giang, rồi Phú Yên. Nhưng được sự ủng hộ của nhân dân, ông đã chống lệnh triều đình, quyết tâm ở lại kháng chiến. Phất cao lá cờ "Bình Tây Đại nguyên soái", hoạt động của nghĩa quân đã củng cố niềm tin của dân chúng, khiến bọn cướp nước và bán nước phải run sợ.'
    f"Output: (Phong trào tị địa, diễn ra, rất sôi nổi){KG_TRIPLE_DELIMITER}(đội nghĩa quân, không chịu, hạ vũ khí){KG_TRIPLE_DELIMITER}(Cuộc khởi nghĩa Trương Định, giành được, thắng lợi)){KG_TRIPLE_DELIMITER}\n"
    f"{KG_TRIPLE_DELIMITER}(Trương Định, theo, cha vào Nam từ hồi nhỏ){KG_TRIPLE_DELIMITER}(Trương Định, là, con của Lãnh binh Trương Cầm){KG_TRIPLE_DELIMITER}\n"
    "EXAMPLE\n"
    "{text}"
    "Output:"
)

PROMPT = PromptTemplate(
    input_variables=["text"], template=_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE
)

prompt_summary_template = """Write a concise summary of the following:


{text}


CONCISE SUMMARY IN VIETNAMESE:"""


class GraphExtractor(BaseModel):
    """Functionality to create graph index."""

    llm: Optional[BaseLLM] = OpenAI(
        openai_api_key="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"
    )
    prompt: PromptTemplate = PROMPT
    graph_type: Type[NetworkxEntityGraph] = NetworkxEntityGraph

    summary_prompt = PromptTemplate(
        template=prompt_summary_template, input_variables=["text"]
    )

    def from_text(self, text: str) -> NetworkxEntityGraph:
        """Create graph index from text."""
        if self.llm is None:
            raise ValueError("llm should not be None")
        graph = self.graph_type()
        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        output = chain.predict(text=text)
        knowledge = parse_triples(output)
        for triple in knowledge:
            graph.add_triple(triple)
        return graph

    def annotate(self, text):
        docs = [Document(page_content=t) for t in [text]]
        summary_chain = load_summarize_chain(
            self.llm, chain_type="stuff", prompt=self.summary_prompt
        )
        text = summary_chain.run(docs)
        graph = self.from_text(text)
        return graph.get_triples()


def run_test():
    context = """Như vậy, đến giữa thế kỉ XIX, Nhật Bản đã lâm vào một cuộc khủng hoảng trầm trọng, đứng trước sự lựa chọn : hoặc tiếp tục duy trì chế độ phong kiến trì trệ, bảo thủ để bị các nước đế quốc xâu xé hoặc tiến hành duy tân, đưa Nhật Bản phát triển theo con đường của các nước tư bản phương Tây. Phong trào đấu tranh chống Sôgun phát triển mạnh vào những năm 60 của thế kỉ XIX đã làm sụp đổ chế độ Mạc phủ. Đó là cuộc Duy tân Minh Trị, được tiến hành trên tất cả các lĩnh vực : chính trị, kinh tế, quân sự, văn hoá - giáo dục... Về chính trị : Nhật hoàng tuyên bố thủ tiêu chế độ Mạc phủ, thành lập chính phủ mới, trong đó đại biểu của tầng lớp quý tộc tư sản hoá đóng vai trò quan trọng, thực hiện quyền bình đẳng giữa các công dân."""
    extractor = GraphExtractor()
    triplets = extractor.annotate(context)
