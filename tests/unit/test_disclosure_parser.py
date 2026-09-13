from supply_chain_alpha.data.disclosures import extract_counterparty_candidates


def test_conservative_text_extraction():
    text = "主要客户：华南汽车股份有限公司\n其他说明。供应商名称：东方设备股份有限公司"
    assert extract_counterparty_candidates(text, "customer") == ["华南汽车股份有限公司"]
    assert extract_counterparty_candidates(text, "supplier") == ["东方设备股份有限公司"]
