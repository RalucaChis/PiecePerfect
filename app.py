import os

from flask import Flask, render_template, request
from flask_paginate import Pagination, get_page_args
from werkzeug.utils import secure_filename

from src.segmentation.utils import generate_mask, crop_around_piece, add_padding_reshape_to_148
from src.segmentation.u_net_network import U_Net, encoder_block, decoder_block, conv_block
from src.edge_matching.siamese_network import SiameseNetwork
from match_estimation import find_top_5_matches
import __main__


setattr(__main__, "U_Net", U_Net)
setattr(__main__, "SiameseNetwork", SiameseNetwork)
setattr(__main__, "encoder_block", encoder_block)
setattr(__main__, "decoder_block", decoder_block)
setattr(__main__, "conv_block", conv_block)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/puzzle_pieces'
app.config['MASKS_FOLDER'] = './static/masks'


@app.route('/home', methods=['GET', 'POST'])
def index():
    return render_template('home.html')


@app.route('/our-mission', methods=['GET', 'POST'])
def our_mission_btn_action():
    return render_template('our-mission.html')


@app.route('/collection', methods=['GET', 'POST'])
def go_to_my_collection_btn_action():
    matches_list = []
    image_rows = []
    is_image_loaded = False
    chosen_image = ''
    if request.method == 'POST':
        if "upload_img" in request.form:
            f = request.files['file']
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
            mask_path = os.path.join(app.config['MASKS_FOLDER'], secure_filename(f.filename))

            generate_mask(img_path, mask_path)
            crop_around_piece(mask_path)
            add_padding_reshape_to_148(mask_path)
            is_image_loaded = True

        elif "find_matches" in request.form:
            selected_checkbox = ""
            for checkbox_id in request.form:
                if checkbox_id != "find_matches":
                    selected_checkbox = checkbox_id
                    chosen_image = selected_checkbox
            if selected_checkbox != "":
                matches_list = find_matches(selected_checkbox)
            else:
                matches_list = []

    page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')
    per_page = 2
    offset = (page - 1) * per_page
    image_list = os.listdir(app.config['UPLOAD_FOLDER'])
    for i in range(0, len(image_list), 6):
        image_rows += [image_list[i:i + 6]]

    total = len(image_rows)
    image_rows = image_rows[offset:offset + per_page]
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')

    return render_template('collection.html',
                           image_rows=image_rows,
                           pagination=pagination,
                           per_page=per_page,
                           page=page,
                           matches_list=matches_list,
                           chosen_image=chosen_image,
                           is_image_loaded=is_image_loaded)


def find_matches(img_name):
    mask_path = os.path.join(app.config['MASKS_FOLDER'], img_name)
    if not os.path.exists(mask_path):
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        generate_mask(img_path, mask_path)
        crop_around_piece(mask_path)
        add_padding_reshape_to_148(mask_path)

    found_matches, probabilities = find_top_5_matches(img_name, app.config['MASKS_FOLDER'])
    print(probabilities)
    if not found_matches:
        found_matches = ['vincent_17.JPG', 'vincent_15.JPG', 'vincent_24.JPG', 'vincent_9.JPG', 'vincent_12.JPG']
    return found_matches


@app.template_filter('length')
def get_length(obj):
    return len(obj)


if __name__ == '__main__':
    app.run(debug=True)
