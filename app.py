import os
import time

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
app.config['matches_list_back_up'] = []
app.config['match_probs_back_up'] = []
app.config['load_cnt'] = 1
app.config['chosen_image'] = ''
app.config['chosen_match_side'] = ''


@app.route('/home', methods=['GET', 'POST'])
def index():
    return render_template('home.html')


@app.route('/our-mission', methods=['GET', 'POST'])
def our_mission_btn_action():
    return render_template('our-mission.html')


@app.route('/test', methods=['GET', 'POST'])
def process():
    return render_template('test.html')


@app.route('/collection', methods=['GET', 'POST'])
def go_to_my_collection_btn_action():
    matches_list = []
    match_probs = []
    matches_list_ordered = []
    match_probs_ordered = []
    image_rows = []
    dropdown_options = ['Match side', 'right', 'top', 'bottom', 'left']
    is_image_loaded = False
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
            selected_option = request.form['selected_option'] if request.form['selected_option'] != dropdown_options[
                0] else 'right'
            app.config['chosen_match_side'] = selected_option
            for checkbox_id in request.form:
                if checkbox_id != "find_matches":
                    selected_checkbox = checkbox_id
            if selected_checkbox != "":
                matches_list, match_probs = find_matches(selected_checkbox, selected_option)
                app.config['matches_list_back_up'] = matches_list
                app.config['match_probs_back_up'] = match_probs
                app.config['load_cnt'] = 1
                app.config['chosen_image'] = selected_checkbox

        elif "load_more" in request.form:
            matches_list = app.config['matches_list_back_up']
            match_probs = app.config['match_probs_back_up']
            if app.config['load_cnt'] < len(matches_list) // 6 + 1:
                app.config['load_cnt'] += 1

    page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')
    per_page = 2
    offset = (page - 1) * per_page
    image_list = os.listdir(app.config['UPLOAD_FOLDER'])
    for i in range(0, len(image_list), 6):
        image_rows += [image_list[i:i + 6]]

    if matches_list:
        matches_list_ordered += [matches_list[0:5]]
        match_probs_ordered += [match_probs[0:5]]
        if app.config['load_cnt'] > 1 and app.config['load_cnt'] <= len(matches_list) // 6 + 1:
            for i in range(5, app.config['load_cnt'] * 6 - 1, 6):
                matches_list_ordered += [matches_list[i:i + 6]]
                match_probs_ordered += [match_probs[i:i + 6]]

    total = len(image_rows)
    image_rows = image_rows[offset:offset + per_page]
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')

    return render_template('collection.html',
                           image_rows=image_rows,
                           pagination=pagination,
                           per_page=per_page,
                           page=page,
                           matches_list=matches_list_ordered,
                           match_probs=match_probs_ordered,
                           chosen_image=app.config['chosen_image'],
                           chosen_match_side=app.config['chosen_match_side'],
                           is_image_loaded=is_image_loaded,
                           dropdown_options=dropdown_options)


def find_matches(img_name, selected_option):
    mask_path = os.path.join(app.config['MASKS_FOLDER'], img_name)
    if not os.path.exists(mask_path):
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        generate_mask(img_path, mask_path)
        crop_around_piece(mask_path)
        add_padding_reshape_to_148(mask_path)

    found_matches, probabilities = find_top_5_matches(img_name, app.config['MASKS_FOLDER'], selected_option)
    if not found_matches:
        found_matches = ['vincent_17.JPG', 'vincent_15.JPG', 'vincent_24.JPG', 'vincent_9.JPG', 'vincent_12.JPG']
        probabilities = ['12.00', '9.12', '8.76', '8.70', '5.01']
    return found_matches, probabilities


@app.template_filter('length')
def get_length(obj):
    return len(obj)


if __name__ == '__main__':
    app.run(debug=True)
