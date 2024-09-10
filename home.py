from flask import Flask, request, render_template,jsonify
from TextSearch import textSearch
import json 
link_bin_file = r"C:\Users\leduc\ai\data\faiss_clipv2_cosine.bin"
json_file = r"C:\Users\leduc\ai\data\id2img_fps.json"
text_search= textSearch(link_bin_file,json_file)
app =Flask(__name__)

@app.route("/")
def main_pages():
    return render_template("home.html")

@app.route("/search", methods=["POST"])
def query_search():
    """this function is designed for taking input queries and getting image"""
    input_queries = request.form.get("query")
    if not input_queries:
        return jsonify({"error": "No input provided"}), 404
    else:
        scores, idx_image, link_paths = text_search.text_search(input_queries, k=10, index=None)
        return jsonify({"scores": scores, "idx_image": idx_image, "link_paths": link_paths})
    

if __name__ =="__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
