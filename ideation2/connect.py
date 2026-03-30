from elasticsearch import Elasticsearch

es = Elasticsearch(
    "https://127.0.0.1:9200",
    basic_auth=("elastic", "dcJwpNq4m6wtDNpdbUhY"),
    verify_certs=False
)

print(es.info())