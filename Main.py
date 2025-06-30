from kivymd.app import MDApp
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Rectangle
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivymd.uix.datatables import MDDataTable
from kivy.metrics import dp
from kivymd.uix.label import MDLabel
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.textfield import MDTextField
from kivy.uix.screenmanager import ScreenManager, Screen
from kivymd.uix.button import MDRaisedButton
from datetime import datetime
from kivymd.uix.boxlayout import MDBoxLayout
from kivy_garden.mapview import MapView, MapMarker
from kivymd.uix.dialog import MDDialog
from kivy.metrics import dp
from plyer import gps
from kivy.config import Config
from kivymd.uix.button import MDFlatButton
from kivy.animation import Animation
from kivy.uix.vkeyboard import VKeyboard
from kivymd.uix.progressbar import MDProgressBar
from kivy.uix.relativelayout import RelativeLayout
from kivymd.uix.selectioncontrol import MDCheckbox
from kivymd.uix.textfield import MDTextField
from kivy.uix.scrollview import ScrollView
from kivymd.uix.card import MDCard


import locale
import cv2
import requests
import math
import os
import joblib
import json
import cvzone
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import time
import threading
import ast

#Window.borderless = True
Window.fullscreen = True

# URL API absensi
API_URL = "https://hrd.asmat.app/api/v2/absen"
Regis_URL = "https://hrd.asmat.app/api/v2/face-recognition/register"

model = YOLO("model/l_versions_3_100.pt")
classNames = ["fake", "real"]

app = FaceAnalysis(providers=['CPUExecutionProvider'])  # Gunakan GPU jika ada
app.prepare(ctx_id=-1)

# Path folder penyimpanan gambar
capture_folder = "./static/captures"
os.makedirs(capture_folder, exist_ok=True)  # Buat folder jika belum ada

data_json = "./Data Pegawai/Json/data_table.json"
os.makedirs(os.path.dirname(data_json), exist_ok=True)

database_json = "./Data Pegawai/Json/database.json"
# Load data wajah dari folder Joblib
joblib_file = "./Data Pegawai/Joblib/database.joblib"

confidence = 0.6
last_seen = {}  # Dictionary untuk menyimpan status wajah terakhir terlihat
face_data = {}
loading_done = False  # Indikator bahwa proses load selesai

# Dictionary untuk menyimpan wajah yang dikenali dan posisinya
recognized_faces = {}

def is_internet_available():
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False


# Load database JSON jika ada
if os.path.exists(database_json):
    with open(database_json, "r") as file:
        try:
            database = json.load(file)
        except json.JSONDecodeError:
            database = []
else:
    database = []

# Cek apakah file joblib ada, jika tidak buat list baru
if os.path.exists(joblib_file):
    database_joblib_data = joblib.load(joblib_file)
else:
    database_joblib_data = []

def load_joblib_data():
    """Fungsi untuk load semua data wajah dari satu file joblib."""
    global face_data, loading_done
    face_data.clear()

    if os.path.exists(joblib_file):
        data = joblib.load(joblib_file)

        if isinstance(data, list):  # Cek apakah data dalam format list
            for entry in data:
                if 'nrp' in entry and 'encodings' in entry:
                    face_data[entry['nrp']] = np.array(entry['encodings'])
                else:
                    print(f"⚠ Data tidak valid: {entry}")
        else:
            print("⚠ Format database.joblib tidak sesuai!")

    else:
        print("⚠ File database.joblib tidak ditemukan!")

    loading_done = True
    print("✅ Semua data wajah telah dimuat!")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_face(img, main_content):
    img = cv2.flip(img, 1)
    faces = app.get(img)
     

    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        frame_width = img.shape[1]
        mirrored_x1 = frame_width - x1
        face_encoding = np.array(face.normed_embedding)

        matched_nrps = []
        for nrp, stored_encoding in face_data.items():
            similarity = cosine_similarity(face_encoding, stored_encoding)
            if similarity > 0.7:
                matched_nrps.append(nrp)
            if not matched_nrps:
                # Jika tidak cocok, simpan info bbox ke recognized_faces dengan label "Unknown"
                recognized_faces["Unknown"] = (mirrored_x1, y1, time.time())

        for matched_nrp in matched_nrps:
            current_time = time.time()

            if matched_nrp not in last_seen:
                last_seen[matched_nrp] = current_time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                image_filename = f"{matched_nrp}_{timestamp}.jpg"
                image_path_local = os.path.join(capture_folder, image_filename)
                cv2.imwrite(image_path_local, img)
                main_content.send_to_server(matched_nrp, image_path_local)
            else:
                if current_time - last_seen[matched_nrp] >= 5:
                    last_seen[matched_nrp] = current_time
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    image_filename = f"{matched_nrp}_{timestamp}.jpg"
                    image_path_local = os.path.join(capture_folder, image_filename)
                    cv2.imwrite(image_path_local, img)
                    main_content.send_to_server(matched_nrp, image_path_local)

            recognized_faces[matched_nrp] = (mirrored_x1, y1, time.time())

            
# Memaksa penggunaan keyboard virtual
Config.set('kivy', 'keyboard_mode', 'systemandmulti')

#Window.size = (540, 960)
Window.size = (1080, 1920)

class Sidebar(BoxLayout):
    def __init__(self, screen_manager, **kwargs):
        super().__init__(**kwargs)

        self.screen_manager = screen_manager  # Simpan ScreenManager utama

        print("🟢 Sidebar menerima ScreenManager dengan layar:", [screen.name for screen in self.screen_manager.screens])
        self.orientation = "vertical"
        self.size_hint = (0.2, 1)
        self.padding = 10
        self.pos_hint = {"top": 1}

        with self.canvas.before:
            Color(0.631, 0.694, 0.909, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)

        self.bind(size=self.update_rect, pos=self.update_rect)

        # Tambahkan profile logo dengan event handler
        self.profile_logo = Image(
            source="assets/profile_icon.png",
            size_hint=(None, 0.2),
            size=(dp(200), dp(200)),
            keep_ratio=True, # Agar rasio tidak berubah
            allow_stretch=True,  # Biar gambar tetap bisa menyesuaikan
            pos_hint={"center_x": 0.5}  # Posisi di tengah atas
        )
        self.profile_logo.bind(on_touch_down=self.on_profile_logo_pressed)
        self.add_widget(self.profile_logo)

        self.remove_logo = Image(
            source="assets/remove_file.png",
            size_hint=(None, 0.2),
            size=(dp(200), dp(200)),
            keep_ratio=True, # Agar rasio tidak berubah
            allow_stretch=True,  # Biar gambar tetap bisa menyesuaikan
            pos_hint={"center_x": 0.5}  # Posisi di tengah atas
        )
        self.remove_logo.bind(on_touch_down=self.delete_files)
        self.add_widget(self.remove_logo)       

        self.fr_text = Image(
            source="assets/FR_TEXT.png",
            size_hint=(None, 0.8),
            size=(dp(100), dp(100)),
            allow_stretch=True,  # Biar gambar tetap bisa menyesuaikan
            keep_ratio=True,  # Agar rasio tidak berubah
            pos_hint = {"center_x": 0.5}
        )
        self.add_widget(self.fr_text)

        self.nrp_checkboxes = {}
        self.dialog_hapus_nrp = None
        self.vkeyboard = None

    def update_rect(self, *args):
        self.rect.size = self.size
        self.rect.pos = self.pos

    def on_profile_logo_pressed(self, instance, touch):
        if self.profile_logo.collide_point(*touch.pos):
            if self.screen_manager.current == "main":

                # *Matikan kamera di MainContent dulu*
                main_screen = self.screen_manager.get_screen("main")
                if hasattr(main_screen, "main_content"):
                    main_screen.main_content.on_leave()

                # *Pindah ke Registration setelah kamera tertutup*
                if "register" in [screen.name for screen in self.screen_manager.screens]:
                    self.screen_manager.current = "register"

                register_screen = self.screen_manager.get_screen("register")
                if hasattr(register_screen, "main_content"):
                    register_screen.main_content.on_enter()

            else:

                # *Matikan kamera di Registration dulu*
                register_screen = self.screen_manager.get_screen("register")
                if hasattr(register_screen, "main_content"):
                    register_screen.main_content.on_leave()

                # *Pindah ke MainContent setelah kamera tertutup*
                self.screen_manager.current = "main"

                main_screen = self.screen_manager.get_screen("main")
                if hasattr(main_screen, "main_content"):
                    main_screen.main_content.on_enter()

    def delete_files(self, instance, touch):
        if not self.remove_logo.collide_point(*touch.pos):
            return

        if not os.path.exists(joblib_file):
            return

        try:
            data = joblib.load(joblib_file)
            nrp_list = [entry["nrp"] for entry in data if "nrp" in entry]
            print("📋 NRP Ditemukan:", nrp_list)
        except Exception as e:
            print("❌ Gagal load joblib:", e)
            nrp_list = []


        content = MDBoxLayout(orientation="vertical", spacing=10, size_hint_y=None, height=dp(600), md_bg_color=(0.945, 0.960, 1, 1))

        self.nrp_checkboxes.clear()

        search_row = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(60), spacing=10)

        self.search_field = MDTextField(
            hint_text="Cari NRP...",
            size_hint_x=0.8,
            font_size=dp(20)
        )

        search_button = MDRaisedButton(
            text="Cari",
            size_hint_x=0.2,
            font_name="assets/GTVCS-Medium",
            md_bg_color=(0.631, 0.694, 0.909, 1),
            on_release=lambda x: self.filter_nrp_checkboxes(self.search_field, self.search_field.text)
        )

        search_row.add_widget(self.search_field)
        search_row.add_widget(search_button)
        content.add_widget(search_row)


        self.nrp_list_layout = BoxLayout(orientation='vertical', spacing=20, size_hint_y=None)
        self.nrp_list_layout.bind(minimum_height=self.nrp_list_layout.setter('height'))

        scroll = ScrollView(size_hint=(1, None),  size=(dp(600), dp(500)),do_scroll_x=False)
        scroll.add_widget(self.nrp_list_layout)
        content.add_widget(scroll)

        for nrp in nrp_list:
            card = MDCard(
                orientation='horizontal',
                size_hint_y=None,
                size_hint_x=None,
                width=dp(500),
                height=dp(70),
                padding=30,
                md_bg_color=(0.901, 0.925, 1, 1),  # background putih untuk card
                shadow_softness=1,
                shadow_offset=(1, 1),
                elevation=1,  # bayangan
                pos_hint={"center_x": 0.5},
                radius=[25, 25, 25, 25],  # sudut bulat
            )

            label = MDLabel(text=nrp, size_hint_x=0.8, bold=True, font_style="H6", valign="middle")
            checkbox = MDCheckbox(size_hint_x=0.2,  color_active=(0, 0, 0, 1), color_inactive=(0.7, 0.7, 0.7, 1),)
            card.add_widget(label)
            card.add_widget(checkbox)
            self.nrp_list_layout.add_widget(card)
            self.nrp_checkboxes[nrp] = (card, checkbox)

        self.dialog_hapus_nrp = MDDialog(
            title="Pilih NRP untuk dihapus",
            type="custom",
            content_cls=content,
            buttons=[
                MDFlatButton(text="Batal",font_name="assets/GTVCS-Medium",  on_release=lambda x: self.dialog_hapus_nrp.dismiss()),
                MDRaisedButton(text="Hapus", md_bg_color=(0.631, 0.694, 0.909, 1), font_name="assets/GTVCS-Medium", on_release=self.delete_selected_nrp)
            ],
            size_hint=(0.8, None),
            height=dp(900),
            md_bg_color=(0.945, 0.960, 1, 1),
            auto_dismiss=False
        )
        self.dialog_hapus_nrp.open()
        self.vkeyboard = None
        self.search_field.bind(focus=self.show_keyboard)
        
    def delete_files(self, instance, touch):
        if not self.remove_logo.collide_point(*touch.pos):
            return

        if not os.path.exists(joblib_file):
            return

        try:
            data = joblib.load(joblib_file)
            nrp_list = [entry["nrp"] for entry in data if "nrp" in entry]
            print("📋 NRP Ditemukan:", nrp_list)
        except Exception as e:
            print("❌ Gagal load joblib:", e)
            nrp_list = []


        content = MDBoxLayout(orientation="vertical", spacing=10, size_hint_y=None, height=dp(600), md_bg_color=(0.945, 0.960, 1, 1))

        self.nrp_checkboxes.clear()

        search_row = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(60), spacing=10)

        self.search_field = MDTextField(
            hint_text="Cari NRP...",
            size_hint_x=0.8,
            font_size=dp(20)
        )

        search_button = MDRaisedButton(
            text="Cari",
            size_hint_x=0.2,
            font_name="assets/GTVCS-Medium",
            md_bg_color=(0.631, 0.694, 0.909, 1),
            on_release=lambda x: self.filter_nrp_checkboxes(self.search_field, self.search_field.text)
        )

        search_row.add_widget(self.search_field)
        search_row.add_widget(search_button)
        content.add_widget(search_row)


        self.nrp_list_layout = BoxLayout(orientation='vertical', spacing=20, size_hint_y=None)
        self.nrp_list_layout.bind(minimum_height=self.nrp_list_layout.setter('height'))

        scroll = ScrollView(size_hint=(1, None),  size=(dp(600), dp(500)),do_scroll_x=False)
        scroll.add_widget(self.nrp_list_layout)
        content.add_widget(scroll)

        for nrp in nrp_list:
            card = MDCard(
                orientation='horizontal',
                size_hint_y=None,
                size_hint_x=None,
                width=dp(800),
                height=dp(70),
                padding=30,
                md_bg_color=(0.901, 0.925, 1, 1),  # background putih untuk card
                shadow_softness=1,
                shadow_offset=(1, 1),
                elevation=1,  # bayangan
                pos_hint={"center_x": 0.5},
                radius=[25, 25, 25, 25],  # sudut bulat
            )

            label = MDLabel(text=nrp, size_hint_x=0.8, bold=True, font_style="H6", valign="middle")
            checkbox = MDCheckbox(size_hint_x=0.2,  color_active=(0, 0, 0, 1), color_inactive=(0.7, 0.7, 0.7, 1),)
            card.add_widget(label)
            card.add_widget(checkbox)
            self.nrp_list_layout.add_widget(card)
            self.nrp_checkboxes[nrp] = (card, checkbox)

        self.dialog_hapus_nrp = MDDialog(
            title="Pilih NRP untuk dihapus",
            type="custom",
            content_cls=content,
            buttons=[
                MDFlatButton(text="Batal",font_name="assets/GTVCS-Medium",  on_release=lambda x: self.dialog_hapus_nrp.dismiss()),
                MDRaisedButton(text="Hapus", md_bg_color=(0.631, 0.694, 0.909, 1), font_name="assets/GTVCS-Medium", on_release=self.delete_selected_nrp)
            ],
            size_hint=(0.8, None),
            height=dp(900),
            md_bg_color=(0.945, 0.960, 1, 1),
            auto_dismiss=False
        )
        self.dialog_hapus_nrp.open()
        self.vkeyboard = None
        self.search_field.bind(focus=self.show_keyboard)
        
    def filter_nrp_checkboxes(self, instance, value):
        value = value.lower()
        self.nrp_list_layout.clear_widgets()

        for nrp, (layout, checkbox) in self.nrp_checkboxes.items():
            if value in nrp.lower():
                self.nrp_list_layout.add_widget(layout)

    def delete_selected_nrp(self, instance):
        selected_nrp = [nrp for nrp, (_,cb) in self.nrp_checkboxes.items() if cb.active]

        if not selected_nrp:
           return

        # Hapus dari joblib
        if os.path.exists(joblib_file):
            data = joblib.load(joblib_file)
            new_data = [entry for entry in data if entry["nrp"] not in selected_nrp]
            joblib.dump(new_data, joblib_file)

        thread = threading.Thread(target=load_joblib_data)
        thread.start()
        
        # Hapus dari JSON
        if os.path.exists(database_json):
            with open(database_json, "r") as f:
                json_data = json.load(f)
            json_data = [entry for entry in json_data if entry["nrp"] not in selected_nrp]
            with open(database_json, "w") as f:
                json.dump(json_data, f, indent=4)

        self.dialog_hapus_nrp.dismiss()

        dialog = MDDialog(
            title="Berhasil",
            text=f"{len(selected_nrp)} data berhasil dihapus.",
            buttons=[MDFlatButton(text="OK", on_release=lambda x: dialog.dismiss())]
        )
        dialog.open()

    def show_keyboard(self, instance, value):
        if value:  # Jika field mendapatkan fokus
            if self.vkeyboard is None:
                self.vkeyboard = VKeyboard()
                self.vkeyboard.size_hint = (1, None)
                self.vkeyboard.height = Window.height * 0.3  # 30% dari tinggi layar
                self.vkeyboard.pos = (0, 0)  # Posisi di bawah layar
                self.vkeyboard.bind(on_textinput=self.on_textinput)
                self.vkeyboard.bind(on_key_down=self.on_key_down)

                Window.add_widget(self.vkeyboard)  # ⬅ tambahkan langsung ke Window
        else:
            if self.vkeyboard:
                Window.remove_widget(self.vkeyboard)
                self.vkeyboard = None
                
    def on_textinput(self, keyboard, text):
        """Menambahkan teks hanya jika valid"""

        # Tentukan MDTextField yang sedang aktif
        active_field = None
        if self.search_field.focus:
            active_field = self.search_field
        if active_field:
            active_field.text += text  # Tambahkan teks ke field aktif

    def on_key_down(self, keyboard, keycode, text, modifiers):
        """Menangani tombol backspace"""
        # Tentukan MDTextField yang sedang aktif
        active_field = None
        if self.search_field.focus:
            active_field = self.search_field

        if active_field:
            if keycode == "backspace":
                active_field.text = active_field.text[:-1]  # Hapus karakter terakhir
            elif keycode == "enter":
                active_field.focus = False  # Sembunyikan keyboard saat Enter ditekan
            elif keycode == "escape":
                active_field.focus = False
      
class MainContent(BoxLayout):
    manual_lat = -6.866641  # Nilai default latitude (cimahi)
    manual_lon = 107.5347632  # Nilai default longitude (cimahi)

    def __init__(self, screen_manager, **kwargs):
        super().__init__(**kwargs)
        self.screen_manager = screen_manager
        self.absensi_tercatat = {}
        self.orientation = 'vertical'
        self.size_hint = (0.8, 1)
        self.padding = dp(10)
        self.last_recognition_time = 0
        self.recognition_interval = 0.5  # detik, atur sesuai kebutuhan
        self.recognition_thread_running = False
        # Panggil hapus_file_capture setiap 600 detik (10 menit)
        Clock.schedule_interval(self.hapus_file_capture, 600)

        with self.canvas.before:
            Color(1, 1, 1, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)

        # Gunakan nilai dari variabel kelas
        self.manual_lat = MainContent.manual_lat
        self.manual_lon = MainContent.manual_lon
        Clock.schedule_once(lambda dt: self.update_location(lat=self.manual_lat, lon=self.manual_lon), 2)
        locale.setlocale(locale.LC_TIME, "id_ID.utf8")
        self.bind(size=self.update_rect, pos=self.update_rect)

        header_layout = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(200), pos_hint={"center_x":0.5, "y":0.1}, padding=10, spacing=10)

        self.company_logo = Image(
            source="assets/tbk.png",
            size_hint=(1, 1)
        )

        # Tambahkan profile logo dengan event handler
        self.sync_logo = Image(
            source="assets/sync_icon.png",
            size_hint=(1, 0.7),
            pos_hint={"center_y":0.5}
        )
        self.sync_logo.bind(on_touch_down=self.on_sync_logo_pressed)

        header_layout.add_widget(self.company_logo)
        header_layout.add_widget(self.sync_logo)
        self.add_widget(header_layout)

        info_container = MDBoxLayout(
            size_hint=(None, None),
            size=(dp(800), dp(80)),
            pos_hint={"center_x": 0.5, "y":0.2},
            md_bg_color=(0.631, 0.694, 0.909, 1),
            radius=[dp(25), dp(25), dp(25), dp(25)],
            padding=[20, 10, 20, 10]
        )

        self.date_label = MDLabel(
            text=self.get_today_date(),
            theme_text_color="Custom",
            text_color=(0.203, 0.2, 0.2, 1),  # Warna putih
            pos_hint={"center_y":0.5},
            halign="left",
            font_size="25sp",
            size_hint_y=None,
            height=dp(80),
        )
        self.date_label.font_size = dp(23)

        location_container = MDBoxLayout(
            size_hint=(None, None),
            size=(dp(400), dp(60)),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            md_bg_color=(0.909, 0.850, 0.760, 1),
            radius=[dp(25), dp(25), dp(25), dp(25)],
            padding=[20, 10, 20, 10]
        )

        self.location_label= MDLabel(
            text=f"Lat: {self.manual_lat}, Lon: {self.manual_lon}",
            theme_text_color="Custom",
            text_color=(0, 0, 0, 1),
            pos_hint={"center_y":0.5},
            halign="center",
            size_hint=(1, None),
            font_name="assets/GTVCS-Medium.ttf",
            size_hint_y=None,
            height=dp(80)
        )
        location_container.add_widget(self.location_label)
        self.location_label.font_size = dp(23)  # Pakai dp() untuk memastikan skala tetap
        location_container.bind(on_touch_down=self.open_map_page)

        font_path = "assets/GTVCS-Medium.ttf"
        self.date_label.font_name = font_path
        self.location_label.font_name = font_path

        info_container.add_widget(self.date_label)
        info_container.add_widget(location_container)
        self.add_widget(info_container)

        self.time_label = MDLabel(
            text="00:00:00",
            theme_text_color="Custom",
            text_color=(0, 0, 0, 1),
            pos_hint={"center_x":0.5, "y":0.7},
            markup=True,
            halign="center",
            size_hint=(None, None),
            size=(dp(200), dp(70)),
            font_name="assets/GTVCS-Medium.ttf"
        )
        self.time_label.font_size = dp(25)
        self.add_widget(self.time_label)

        self.time_label.font_name = font_path
        Clock.schedule_interval(self.update_time, 0.5)

        # Layout untuk kamera dan instruksi di dalamnya
        self.camera_layout = RelativeLayout(
            size_hint=(1,1),
            pos_hint={"center_x": 0.5, "y":0.5}
        )

        # Kamera Webcam
        self.camera_display = Image(size_hint=(1, 1), pos_hint={"center_x": 0.5, "center_y": 0.5})
        self.instruction_container = MDBoxLayout(
            size_hint=(None, None),
            size=(dp(600), dp(60)),
            pos_hint={"center_x": 0.5, "y": 0.15},
            md_bg_color=(0.219, 0.262, 0.411, 1),
            radius=[dp(10), dp(10), dp(10), dp(10)],
            padding=[10, 5, 10, 5]
        )

        self.instruction_main = MDLabel(
            text="Fokuskan wajah Anda pada kamera",
            theme_text_color="Custom",
            text_color=(1, 1, 1, 1),
            pos_hint={"center_y":0.5},
            halign="center",
            size_hint=(1, None),
            font_name="assets/GTVCS-Medium.ttf",
            height=dp(80)
        )
        self.instruction_main.font_size = dp(20)

        self.instruction_container.add_widget(self.instruction_main)
        self.camera_layout.add_widget(self.camera_display)
        self.camera_layout.add_widget(self.instruction_container)
        self.instruction_main.font_name = font_path
        self.add_widget(self.camera_layout)

        # Filter
        filter_layout = BoxLayout(orientation="horizontal", size_hint_y=None, height=dp(100), pos_hint={"center_y":0.4}, padding=(10, 10))
        # filter nama
        self.name_filter = MDTextField(
            hint_text="Cari NRP...",
            size_hint_x=0.2,
            font_size="20sp",
            font_name="assets/GTVCS-Medium",
            on_text_validate=self.filter_table
        )

        self.time_filter_btn = MDRaisedButton(
            text="Semua", size_hint_x=1, md_bg_color=(0.631, 0.694, 0.909, 1), text_color=(0.203, 0.2, 0.2, 1),
            font_size="20sp",font_name="assets/GTVCS-Medium", on_release=self.open_time_menu
        )

        # dropdown waktu
        self.time_menu = MDDropdownMenu(
            caller=self.time_filter_btn,
            items=[
                {"viewclass": "OneLineListItem", "text": "Semua",
                 "theme_text_color": "Custom", "text_color": (0, 0, 0, 1), "font_style": "H6",
                 "on_release": lambda x="Semua": self.set_time_filter(x)},
                {"viewclass": "OneLineListItem", "text": "07:30 - 12:00",
                 "theme_text_color": "Custom", "text_color": (0, 0, 0, 1), "font_style": "H6",
                 "on_release": lambda x="07:30 - 12:00": self.set_time_filter(x)},
                {"viewclass": "OneLineListItem", "text": "12:00 - 18:30",
                 "theme_text_color": "Custom", "text_color": (0, 0, 0, 1), "font_style": "H6",
                 "on_release": lambda x="12:00 - 18:30": self.set_time_filter(x)}
            ],
            width_mult=3
        )

        self.cap = None
        filter_layout.add_widget(self.name_filter)
        filter_layout.add_widget(self.time_filter_btn)
        self.add_widget(filter_layout)

        self.original_data = []
        self.filtered_data = []
        self.load_table_data()
        self.vkeyboard = None
        self.name_filter.bind(focus=self.show_keyboard)
        self.lat_input = None  # 🟢 Pastikan atribut ada, meskipun None
        self.lon_input = None  # 🟢 Pastikan atribut ada, meskipun None

        Clock.schedule_interval(self.check_internet_connection, 10)  # setiap 10 detik


    def show_no_internet_dialog(self):
        if hasattr(self, 'no_internet_dialog') and self.no_internet_dialog:
            return  # Jangan buka lagi kalau sudah terbuka
        
        no_internet_container = MDBoxLayout(
            size_hint=(None, None),
            size=(dp(500), dp(80)),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            md_bg_color=(0.921, 0.364, 0.380, 1),
            radius=[dp(25), dp(25), dp(25), dp(25)],
            padding=[20, 10, 20, 10]
        )
        no_internet_label = MDLabel (text="[size=25]Jaringan internet anda terputus, silahkan periksa kembali.[/size]", font_name="assets/GTVCS-Medium.ttf", halign="center", pos_hint={"center_x":0.5, "center_y":0.5}, markup=True)
        no_internet_container.add_widget(no_internet_label)

        no_internet_content=MDBoxLayout(orientation='vertical', spacing=10, padding=30, md_bg_color=(0, 0, 0, 0))
        no_internet_content.add_widget(Image(source="assets/no_internet_icon.png", size_hint=(1, None), height=250, pos_hint={"center_x":0.5, "y":0.4}))
        no_internet_content.add_widget(no_internet_container)
        self.no_internet_dialog = MDDialog(
            
            type="custom",
            height=50,
            radius=[dp(20), dp(20), dp(20), dp(20)],
            content_cls=no_internet_content,
            
        )
        self.no_internet_dialog.open()


    def check_internet_connection(self, dt):
        if not is_internet_available():
            if not hasattr(self, 'no_internet_dialog') or not self.no_internet_dialog:
                self.show_no_internet_dialog()
        else:
            if hasattr(self, 'no_internet_dialog') and self.no_internet_dialog:
                self.no_internet_dialog.dismiss()
                self.no_internet_dialog = None

    def on_enter(self):
        """Aktifkan kamera hanya jika masuk ke MainContent"""

        # Jalankan proses loading di thread terpisah
        thread = threading.Thread(target=load_joblib_data)
        thread.start()

        self.cap = CameraSingleton.get_instance(screen_name="MainContent")
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        Clock.schedule_interval(self.update_camera, 1.0 / 30.0)

    def on_leave(self):
        """Matikan kamera saat keluar dari MainContent"""
        self.stop_camera()
        CameraSingleton.release(screen_name="MainContent")

    def stop_camera(self):
        """Hentikan update kamera"""
        Clock.unschedule(self.update_camera)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def update_location(self, **kwargs):
        """Memperbarui lokasi pada tampilan aplikasi"""
        lat = kwargs.get('lat', self.manual_lat)
        lon = kwargs.get('lon', self.manual_lon)

        print(f"Update lokasi ke: {lat}, {lon}")  # Debugging

        # Dapatkan nama jalan berdasarkan latitude dan longitude
        address = self.get_address_from_lat_lon(lat, lon)

        # Paksa perubahan label
        self.location_label.text = address
        self.location_label.texture_update()  # Memastikan perubahan langsung terlihat di UI

        # Jika ada MapView, perbarui posisi marker
        if hasattr(self, 'mapview'):
            self.mapview.center_on(lat, lon)
            if hasattr(self, 'marker'):
                self.mapview.remove_widget(self.marker)
            self.marker = MapMarker(lat=lat, lon=lon)
            self.mapview.add_widget(self.marker)

        print(f"Marker diperbarui ke: {lat}, {lon}")  # Debugging

    def get_address_from_lat_lon(self, lat, lon):
        try:
            url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=18&addressdetails=1"
            headers = {"User-Agent": "AttendanceHRIS/1.0(muthianurul6@gmail.com)"}  # Tambahkan User-Agent agar tidak ditolak oleh server
            response = requests.get(url, headers=headers, timeout=5)
            data = response.json()

            if "address" in data:
                address_components = data["address"]

                # Ambil nilai desa, kota, dan provinsi, jika tidak ada, gunakan string kosong
                desa = address_components.get("village", "")
                kota = address_components.get("city", "")
                provinsi = address_components.get("state", "")

                if lat == -6.866641 and lon == 107.5347632:
                    desa = "Cipageran"

                # Gabungkan hanya komponen yang memiliki nilai
                address_parts = [part for part in [desa, kota, provinsi] if part]
                return ", ".join(address_parts) if address_parts else "Alamat tidak ditemukan"

            return "Alamat tidak ditemukan"
        except requests.exceptions.RequestException as e:
            print(f"Error mengambil data lokasi: {e}")
            return "Lokasi Tidak Diketahui"

    def on_status(self, stype, status):
        """Menampilkan status GPS"""
        print(f"Status GPS: {stype}, {status}")

    def open_map_page(self, instance, touch):
        """Menampilkan peta dengan lokasi yang dapat diubah secara manual"""
        if self.location_label.collide_point(*touch.pos):
            self.clear_widgets()

            # Layout utama untuk menyusun elemen secara vertikal
            main_layout = BoxLayout(
                orientation="vertical",
                spacing=20,
                padding=10,
                size_hint=(1, 1)
            )

            # Layout untuk MapView agar berada di bagian tengah
            map_layout = BoxLayout(
                size_hint=(1, 3),  # Sesuaikan tinggi agar peta tidak terlalu besar
                pos_hint={"center_x":0.5}
            )
            self.mapview = MapView(
                zoom=15,
                lat=self.manual_lat,
                lon=self.manual_lon,
                size_hint=(1, 1)
            )
            map_layout.add_widget(self.mapview)

            # Tambahkan marker lokasi
            self.marker = MapMarker(lat=self.manual_lat, lon=self.manual_lon)
            self.mapview.add_widget(self.marker)

            # Layout untuk input latitude dan longitude
            input_layout = BoxLayout(
                orientation="vertical",
                size_hint=(1, 0.3),
                width=dp(1000),
                spacing=20
            )

            self.lat_input_label = MDBoxLayout(
                size_hint=(0.9, None),
                size = (dp(1000), dp(70)),
                pos_hint={"center_x": 0.5},
                md_bg_color=(0.905, 0.905, 0.909, 1),
                radius=[dp(10), dp(10), dp(10), dp(10)],
                padding=10
            )

            self.lat_input = MDTextField(
                hint_text="Latitude",
                text=str(self.manual_lat),
                size_hint_x=1,
                font_size=dp(20),
            )
            self.lat_input_label.add_widget(self.lat_input)

            self.lon_input_label = MDBoxLayout(
                size_hint=(0.9, None),
                size = (dp(1000), dp(70)),
                pos_hint={"center_x": 0.5},
                md_bg_color=(0.905, 0.905, 0.909, 1),
                radius=[dp(10), dp(10), dp(10), dp(10)],
                padding=10
            )

            self.lon_input = MDTextField(
                hint_text="Longitude",
                text=str(self.manual_lon),
                size_hint_x=1,
                font_size=dp(20),
            )

            self.lon_input_label.add_widget(self.lon_input)

            # Tambahkan event listener untuk memunculkan keyboard
            self.lat_input.bind(focus=self.show_keyboard)
            self.lon_input.bind(focus=self.show_keyboard)

            input_layout.add_widget(self.lat_input_label)
            input_layout.add_widget(self.lon_input_label)

            # Layout untuk tombol "Kembali" dan "Perbarui"
            button_layout = BoxLayout(
                size_hint=(1, 0.1),  # 15% dari layar untuk tombol
                pos_hint={"y": 0},
                spacing=35
            )

            back_button = MDRaisedButton(
                text="Kembali",
                size_hint_x=0.5,
                md_bg_color=(0.203, 0.2, 0.2, 1),
                font_name="assets/GTVCS-Medium",
                font_size=dp(20),
                on_release=self.back_to_main
            )

            update_button = MDRaisedButton(
                text="Perbarui",
                size_hint_x=0.5,
                text_color=(0, 0, 0, 1),
                md_bg_color=(0.631, 0.694, 0.909, 1),
                font_name="assets/GTVCS-Medium",
                font_size=dp(20),
                on_release=self.update_marker_location
            )

            button_layout.add_widget(back_button)
            button_layout.add_widget(update_button)

            # Susun elemen-elemen ke dalam main layout
            main_layout.add_widget(map_layout)
            main_layout.add_widget(input_layout)
            main_layout.add_widget(button_layout)

            self.add_widget(main_layout)
            
    def show_keyboard(self, instance, value):
        """Menampilkan atau menyembunyikan keyboard virtual ketika MDTextField mendapatkan fokus"""
        if value:  # Jika field mendapatkan fokus
            if self.vkeyboard is None:
                self.vkeyboard = VKeyboard()
                self.vkeyboard.size_hint = (1, None)
                self.vkeyboard.height = Window.height * 0.30  # Gunakan 35% tinggi layar
                self.vkeyboard.bind(on_textinput=self.on_textinput)  # Menangani teks biasa
                self.vkeyboard.bind(on_key_down=self.on_key_down)  # Menangani tombol khusus seperti backspace
                self.add_widget(self.vkeyboard)
                print("Keyboard virtual ditampilkan")
        else:  # Jika fokus hilang, sembunyikan keyboard
            if self.vkeyboard:
                self.remove_widget(self.vkeyboard)
                self.vkeyboard = None
                print("Keyboard virtual disembunyikan")

    def on_textinput(self, keyboard, text):
        """Menambahkan teks hanya jika valid"""

        # Tentukan MDTextField yang sedang aktif
        active_field = None
        if self.lat_input and self.lat_input.focus:
            active_field = self.lat_input
        elif self.lon_input and self.lon_input.focus:
            active_field = self.lon_input
        elif self.name_filter.focus:
            active_field = self.name_filter

        if active_field:
            active_field.text += text  # Tambahkan teks ke field aktif

    def on_key_down(self, keyboard, keycode, text, modifiers):
        """Menangani tombol backspace"""
        # Tentukan MDTextField yang sedang aktif
        active_field = None
        if self.lat_input and self.lat_input.focus:
            active_field = self.lat_input
        elif self.lon_input and self.lon_input.focus:
            active_field = self.lon_input
        elif self.name_filter.focus:
            active_field = self.name_filter

        if active_field:
            if keycode == "backspace":
                active_field.text = active_field.text[:-1]  # Hapus karakter terakhir
            elif keycode == "enter":
                active_field.focus = False  # Sembunyikan keyboard saat Enter ditekan
            elif keycode == "escape":
                active_field.focus = False

    def update_marker_location(self, instance):
        """Memperbarui marker di MapView berdasarkan input latitude dan longitude"""
        try:
            new_lat = float(self.lat_input.text)
            new_lon = float(self.lon_input.text)

            # Perbarui variabel kelas
            MainContent.manual_lat = new_lat
            MainContent.manual_lon = new_lon

            self.manual_lat = new_lat
            self.manual_lon = new_lon

            self.mapview.center_on(new_lat, new_lon)

            if hasattr(self, 'marker'):
                self.mapview.remove_widget(self.marker)

            self.marker = MapMarker(lat=new_lat, lon=new_lon)
            self.mapview.add_widget(self.marker)

            self.location_label.text = f"Lat: {new_lat}, Lon: {new_lon}"
            print(f"Marker diperbarui ke: {new_lat}, {new_lon}")  # Debugging

        except ValueError:
            print("Masukkan koordinat yang valid!")

    def update_marker_location(self, instance):
        """Memperbarui marker di MapView berdasarkan input latitude dan longitude"""
        try:
            new_lat = float(self.lat_input.text)
            new_lon = float(self.lon_input.text)

            # Perbarui variabel kelas
            MainContent.manual_lat = new_lat
            MainContent.manual_lon = new_lon

            self.manual_lat = new_lat
            self.manual_lon = new_lon

            self.mapview.center_on(new_lat, new_lon)

            if hasattr(self, 'marker'):
                self.mapview.remove_widget(self.marker)

            self.marker = MapMarker(lat=new_lat, lon=new_lon)
            self.mapview.add_widget(self.marker)

            self.location_label.text = f"Lat: {new_lat}, Lon: {new_lon}"
            print(f"Marker diperbarui ke: {new_lat}, {new_lon}")  # Debugging

            # Menampilkan notifikasi menggunakan MDDialog
            self.show_update_success_dialog()

        except ValueError:
            print("Masukkan koordinat yang valid!")

    def show_update_success_dialog(self):
        """Menampilkan dialog notifikasi setelah lokasi diperbarui"""
        label_container = MDBoxLayout(
            size_hint=(None, None),
            size=(dp(500), dp(60)),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            md_bg_color=(0.564, 0.874, 0.654, 1),
            radius=[dp(25), dp(25), dp(25), dp(25)],
            padding=[20, 10, 20, 10]
        )
        success_label = MDLabel (text="Lokasi berhasil diperbarui", font_name="assets/GTVCS-Medium.ttf", halign="center", pos_hint={"center_x":0.5, "center_y":0.5})
        label_container.add_widget(success_label)

        content=MDBoxLayout(orientation='vertical', spacing=10, padding=30, md_bg_color=(0, 0, 0, 0))
        content.add_widget(Image(source="assets/complete_icon.png", size_hint=(1, None), height=250, pos_hint={"center_x":0.5, "y":0.4}))
        content.add_widget(label_container)
        dialog = MDDialog(
            type="custom",
            height=50,
            radius=[dp(20), dp(20), dp(20), dp(20)],
            content_cls=content,
            buttons=[
                MDRaisedButton(
                    text="Tutup",
                    md_bg_color=(0.631, 0.694, 0.909, 1),
                    text_color=(0.203, 0.2, 0.2, 1),
                    on_release=lambda x: dialog.dismiss()
                )
            ]
        )
        dialog.open()
        Clock.schedule_once(lambda x: dialog.dismiss(), 10)

    def back_to_main(self, instance):
        self.clear_widgets()

        # Gunakan nilai dari variabel kelas
        self.manual_lat = MainContent.manual_lat
        self.manual_lon = MainContent.manual_lon

        # Kembali ke halaman utama
        self.__init__(self.screen_manager) # Sesuaikan dengan nama halaman utama yang benar
        self.on_enter()
        # Perbarui tampilan dengan nilai yang sudah diperbarui
        self.update_location(lat=self.manual_lat, lon=self.manual_lon)

    def on_sync_logo_pressed(self, instance, touch):
        # URL API dasar
        base_url = "https://hrd.asmat.app/api/v2/face-recognition"

        # Lokasi penyimpanan file JSON dan Joblib
        json_path = "./Data Pegawai/Json/database.json"
        joblib_path = "./Data Pegawai/Joblib/database.joblib"

        # Pastikan folder tersedia
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        os.makedirs(os.path.dirname(joblib_path), exist_ok=True)

        if self.sync_logo.collide_point(*touch.pos):
            # Ambil semua data dari seluruh halaman
            all_data = []
            page = 1
            while True:
                response = requests.get(f"{base_url}?page={page}")
                if response.status_code != 200:
                    print(f"❌ Gagal mengambil data dari halaman {page}")
                    break

                data = response.json()
                if "data" not in data or not data["data"]:
                    break

                all_data.extend(data["data"])
                if page >= data.get("total_pages", 1):
                    break
                page += 1

            if not all_data:
                dialog = MDDialog(text="[size=25]Data tidak ditemukan.[/size]",
                                buttons=[MDFlatButton(text="[size=18]OK[/size]",
                                                        on_release=lambda x: dialog.dismiss())])
                dialog.open()
                return

            extracted_data = []
            for item in all_data:
                nrp = item.get("nrp")
                face_encoding_str = item.get("face_encoding")
                try:
                    face_encoding = ast.literal_eval(face_encoding_str) if face_encoding_str else []
                except:
                    face_encoding = []
                extracted_data.append({
                    "nrp": nrp,
                    "encodings": face_encoding
                })

            # Simpan ke file JSON
            with open(json_path, "w") as json_file:
                json.dump(extracted_data, json_file, indent=4)

            # Simpan ke file Joblib
            joblib.dump(extracted_data, joblib_path)

            print(f"✅ Data berhasil disimpan ke:\n- {json_path}\n- {joblib_path}")
            thread = threading.Thread(target=load_joblib_data)
            thread.start()

            # Menampilkan notifikasi sukses
            success_content = MDBoxLayout(orientation='vertical', spacing=20, md_bg_color=(0, 0, 0, 0))
            success_content.add_widget(Image(
                source="assets/complete_icon.png",
                size_hint=(1, None),
                height=200,
                pos_hint={"center_x": 0.5}
            ))
            success_container = MDBoxLayout(
                size_hint=(None, None),
                size=(dp(500), dp(60)),
                pos_hint={"center_x": 0.5},
                md_bg_color=(0.564, 0.874, 0.654, 1),
                radius=[dp(25)] * 4,
                padding=[20, 10, 20, 10]
            )
            success_label = MDLabel(
                text="[size=25]Sinkronisasi data berhasil![/size]",
                font_name="assets/GTVCS-Medium.ttf",
                halign="center",
                markup=True
            )
            success_container.add_widget(success_label)
            success_content.add_widget(success_container)

            self.success_dialog = MDDialog(
                type="custom",
                height=80,
                radius=[dp(20)] * 4,
                content_cls=success_content,
                buttons=[MDFlatButton(
                    text="[size=18]Tutup[/size]",
                    on_release=lambda x: self.success_dialog.dismiss()
                )]
            )
            self.success_dialog.open()
            Clock.schedule_once(lambda x: self.success_dialog)


    def update_pagination_font(self, *args):
        if self.table:
            # Ambil elemen pagination
            pagination = self.table.pagination

            # Pastikan pagination ada sebelum mengubah font
            if pagination:
                # Ubah ukuran font pada label "Rows per page"
                if "label_rows_per_page" in pagination.ids:
                    pagination.ids.label_rows_per_page.font_size = "20sp"
                    pagination.ids.label_rows_per_page.font_name = "assets/GTVCS-Medium.ttf"

                # Ubah ukuran font pada dropdown jumlah baris per halaman
                if "drop_item" in pagination.ids:
                    pagination.ids.drop_item.font_size = "20sp"
                    pagination.ids.drop_item.font_name = "assets/GTVCS-Medium.ttf"

    def create_table(self):
        if hasattr(self, 'table'):
            self.remove_widget(self.table)

        self.table = MDDataTable(
            size_hint=(None, None),
            width=dp(840),
            height=dp(640),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            use_pagination=True,
            column_data=[
                ("[font=assets/GTVCS-Medium.ttf][size=25]NRP[/size][/font]", dp(50)),
                ("[font=assets/GTVCS-Medium.ttf][size=25]Status[/size][/font]", dp(70)),
                ("[font=assets/GTVCS-Medium.ttf][size=25]Waktu[/size][/font]", dp(50))
            ],
            row_data=[
                (f"[size=21]{nrp}[/size]", f"[size=21]{status}[/size]", f"[size=21]{waktu}[/size]")
                for nrp, status, waktu in self.filtered_data
            ]

        )
        self.add_widget(self.table)
        Clock.schedule_once(lambda dt: self.update_pagination_font(), 0.1)


    def add_to_table(self, nrp, status, waktu):
        """Menambahkan data absensi ke tabel dengan ukuran teks yang disesuaikan"""
        new_entry = (
            f"{nrp}",
            f"{status}",
            f"{waktu}"
        )
        self.original_data.insert(0, new_entry)  # Tambah data baru dengan markup
        self.filtered_data = self.original_data.copy()
        self.create_table()  # Perbarui tampilan tabel
        Clock.schedule_once(lambda dt: self.save_table_data(), 0)

    def save_table_data(self):
        """Menyimpan data tabel dan tanggal terakhir ke dalam file JSON"""
        data_to_save = {
            "tanggal": datetime.today().strftime("%Y-%m-%d"),  # Simpan tanggal hari ini
            "absensi": self.original_data  # Simpan data absensi
        }

        with open(data_json, "w") as file:
            json.dump(data_to_save, file, indent=4)

    def load_table_data(self):
        """Memuat data tabel dari file JSON jika masih dalam hari yang sama, reset jika sudah berganti hari"""
        try:
            with open(data_json, "r") as file:
                data = json.load(file)

            last_saved_date = data.get("tanggal", "")  # Ambil tanggal terakhir
            today_date = datetime.today().strftime("%Y-%m-%d")  # Tanggal hari ini

            if last_saved_date == today_date:
                self.original_data = data.get("absensi", [])
                self.filtered_data = self.original_data.copy() # Debugging
            else:
                print("🔄 Hari baru terdeteksi, data absensi direset!")
                self.original_data = []
                self.filtered_data = []
                self.save_table_data()  # Simpan tabel kosong untuk hari baru

            self.create_table()  # Perbarui tampilan tabel

        except (FileNotFoundError, json.JSONDecodeError):
            print("⚠ Tidak ada data sebelumnya atau file rusak.")
            self.original_data = []
            self.filtered_data = []
            self.create_table()

    def process_server_response(self, nrp, response_data):
        """Menangani respons dari server dan menambahkan data ke tabel."""

        message = response_data.get("message", "Unknown Status")
        waktu_full = response_data.get("data", {}).get("waktu", "Unknown Time")

        # Konversi waktu ke format jam (HH:MM:SS)
        try:
            waktu_obj = datetime.strptime(waktu_full, "%Y-%m-%d %H:%M:%S")
            waktu = waktu_obj.strftime("%H:%M:%S")
        except ValueError:
            waktu = "Unknown Time"

        if "sudah check" in message.lower():
            self.add_to_table(nrp, message, waktu)  # Hapus pengecekan absensi_tercatat

        # Jika berhasil check-in/check-out, selalu tambahkan
        elif "check in berhasil" in message.lower() or "check out berhasil" in message.lower():
            self.add_to_table(nrp, message, waktu)

    def filter_table(self, instance=None):
        name_filter = self.name_filter.text.lower()
        time_filter = self.time_filter_btn.text

        self.filtered_data = [
            row for row in self.original_data
            if
            (name_filter in row[0].lower()) and (time_filter == "Semua" or self.is_time_in_range(row[2], time_filter))
        ]
        self.create_table()  # Perbarui tabel dengan data yang sudah difilter

    def open_time_menu(self, instance):
        self.time_menu.open()

    def set_time_filter(self, time_range):
        self.time_filter_btn.text = time_range
        self.time_menu.dismiss()
        self.filter_table()

    def is_time_in_range(self, time_str, time_range):
        """
        Memeriksa apakah waktu (time_str) berada dalam rentang waktu (time_range).
        time_str: Waktu dalam format "HH:MM:SS".
        time_range: Rentang waktu dalam format "HH:MM - HH:MM" atau "Semua".
        """
        if time_range == "Semua":
            return True

        # Pisahkan rentang waktu menjadi waktu mulai dan waktu selesai
        start_time_str, end_time_str = time_range.split(" - ")

        # Konversi waktu ke objek datetime untuk perbandingan
        time_format = "%H:%M:%S"
        try:
            time_obj = datetime.strptime(time_str, time_format).time()
            start_time_obj = datetime.strptime(start_time_str, "%H:%M").time()
            end_time_obj = datetime.strptime(end_time_str, "%H:%M").time()
        except ValueError:
            print(f"Format waktu tidak valid: time_str={time_str}, time_range={time_range}")
            return False

        # Periksa apakah waktu berada dalam rentang
        return start_time_obj <= time_obj <= end_time_obj

    def get_today_date(self):
        today = datetime.today()
        return today.strftime("%A, %d %B %Y")

    def update_rect(self, *args):
        self.rect.size = self.size
        self.rect.pos = self.pos

    def update_camera(self, dt):

        global recognized_faces
        success, img = self.cap.read()
        img = cv2.flip(img, 1)
        frame_width = img.shape[1]
        img_for_detection = cv2.flip(img, 1)
        results = model(img, stream=True, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                if conf > confidence:
                    color = (0, 255, 0) if classNames[cls] == 'real' else (0, 0, 255)
                    cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)

                    if classNames[cls] == 'real':
                        now = time.time()
                        if not self.recognition_thread_running and (now - self.last_recognition_time > self.recognition_interval):
                            self.recognition_thread_running = True
                            self.last_recognition_time = now

                            def recognition_done(*a):
                                self.recognition_thread_running = False

                            def thread_func():
                                recognize_face(img_for_detection.copy(), self)
                                Clock.schedule_once(recognition_done)
                            threading.Thread(target=thread_func).start()

                        current_time = time.time()
                        recognized_faces = {nrp: (x, y, t) for nrp, (x, y, t) in recognized_faces.items() if
                                            current_time - t < 2}

                        # **Tampilkan teks NRP dari dictionary `recognized_faces`**
                        for nrp, (x1, y1, _) in recognized_faces.items():
                            label = "Tidak Dikenal" if nrp == "Unknown" else f"NRP: {nrp}"
                            color = (0, 0, 255) if nrp == "Unknown" else (0, 255, 0)
                            mirrored_text_x = frame_width - x1
                            cvzone.putTextRect(img, label, (max(0, mirrored_text_x), max(100, y1)), scale=1.5, thickness=2, colorR=color)

            # 🔥 Crop Bagian Tengah ke 850x900
        h, w, _ = img.shape
        x_center, y_center = w // 2, h // 2
        crop_w, crop_h = 960, 1280  # Ukuran portrait
        x1 = max(0, x_center - crop_w // 2)
        x2 = min(w, x_center + crop_w // 2)
        y1 = max(0, y_center - crop_h // 2)
        y2 = min(h, y_center + crop_h // 2)
        img = img[y1:y2, x1:x2]

        # Konversi ke RGB & Kivy Texture
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.flip(img, 0) 
        texture = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='rgb')
        texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='ubyte')

        self.camera_display.texture = texture

    def send_to_server(self, nrp, image_path_local, callback=None):
        """Mengirim data ke server dalam thread, lalu menjalankan callback dengan respons."""

        def thread_task():
            """Fungsi yang berjalan dalam thread."""
            
            with open(image_path_local, "rb") as image_file:
                files = {"foto": (os.path.basename(image_path_local), image_file, "image/jpeg")}
                data = {"nrp": str(nrp)}

                try:
                    response = requests.post(API_URL, data=data, files=files)

                    if response.status_code == 200:
                        response_data = response.json()
                        print(response_data)

                        # Jika ada callback, jalankan dengan data respons
                        Clock.schedule_once(lambda dt: self.process_server_response(nrp, response_data))

                    else:
                        Clock.schedule_once(lambda dt: self.show_server_error_dialog(f"Gagal mengirim data {nrp}. Status: {response.status_code}"))

                except Exception as e:
                     Clock.schedule_once(lambda dt: self.show_server_error_dialog(f"Error saat mengirim data ke server: {e}"))
            
        # Jalankan dalam thread
        thread = threading.Thread(target=thread_task)
        thread.start()

    def show_server_error_dialog(self, message):
        dialog = MDDialog(
            title="Kesalahan Server",
            text=message,
            buttons=[
                MDFlatButton(
                    text="OK",
                    on_release=lambda x: dialog.dismiss()
                )
            ]
        )
        dialog.open()

    def update_time(self, dt):
        """Memperbarui label waktu secara real-time"""
        now = datetime.now().strftime("%H:%M:%S")
        self.time_label.text = now

    def hapus_file_capture(self, *args):
        """Hapus file yang lebih dari 10 menit di folder ./static/captures."""
        now = time.time()
        try:
            for filename in os.listdir(capture_folder):
                file_path = os.path.join(capture_folder, filename)
                if os.path.isfile(file_path):
                    # Hitung usia file
                    file_age = now - os.path.getmtime(file_path)
                    try:
                        os.remove(file_path)
                        print(f"🗑 File {file_path} dihapus (usia: {file_age:.2f} detik).")
                    except Exception as e:
                        print(f"❌ Gagal menghapus {file_path}: {e}")
        except Exception as e:
            print(f"❌ Gagal menghapus file: {e}")

class Registration(Screen):  # Ganti BoxLayout dengan RelativeLayout
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = RelativeLayout()
        self.add_widget(self.layout)
        self.size_hint = (0.8, 1)

        # Tambahkan company_logo di atas regist_graphic
        self.company_logo = Image(
            source="assets/tbk_reg.png",
            size_hint=(1, None),
            height=dp(150),
            keep_ratio = True,  # Agar rasio tidak berubah
            allow_stretch = True,  # Biar gambar tetap bisa menyesuaikan
            pos_hint={"center_x":0.5, "top":0.95}
        )

        # Tambahkan regist_graphic sebagai background
        self.regist_graphic = Image(
            source="assets/regist_graphic.png",
            size_hint=(1, None),  # Ukuran penuh
            height=dp(1700),
            keep_ratio=True,  # Agar rasio tidak berubah
            allow_stretch=True,  # Biar gambar tetap bisa menyesuaikan
            pos_hint={"center_x": 0.5, "bottom":1}  # Posisi tengah
        )

        # Container untuk input nama
        self.registration_text_container = MDBoxLayout(
            orientation='vertical',
            size_hint=(None, None),
            size=(dp(700), dp(100)),
            pos_hint={"center_x": 0.5, "y": 0.6},
            md_bg_color=(0.909, 0.850, 0.760, 1),
            radius=[dp(25), dp(25), dp(25), dp(25)],
            padding=10
        )

        # Tambahkan registration_label
        self.registration_label = MDLabel(
            text="Proses registrasi, mohon arahkan wajah pada kamera",
            halign="center",
            theme_text_color="Custom",
            text_color=(0, 0, 0, 1),
            font_name="assets/GTVCS-Medium.ttf",
            size_hint_y=None,
            height=dp(40),
        )
        self.registration_label.font_size = dp(20)  # Paksa ukuran font lebih besar

        self.registration_progressbar = MDProgressBar(
            size_hint_x=1,
            height=dp(10),
            color=(0.219, 0.262, 0.411, 1),
            radius=[dp(15), dp(15), dp(15), dp(15)],
            value=0
        )
       
        self.registration_text_container.add_widget(self.registration_progressbar)
        self.registration_text_container.add_widget(self.registration_label)
        self.registration_text_container.opacity = 0

        # Layout untuk input nama dan NRP
        self.input_layout = BoxLayout(
            orientation="vertical",
            size_hint_y=None,
            height=dp(200),
            padding=20,
            spacing=50,
            pos_hint={"center_x": 0.5, "y":0.26}
        )

        # Container untuk input NRP
        self.nrp_container = MDBoxLayout(
            size_hint=(None, None),
            size=(dp(560), dp(110)),
            pos_hint={"center_x": 0.5},
            md_bg_color=(0.909, 0.850, 0.760, 1),
            radius=[dp(25), dp(25), dp(25), dp(25)],
            padding=10
        )
        self.nrp_input = MDTextField(
            hint_text="Masukkan 10 digit NRP Anda",
            size_hint_x=None,
            width=dp(530),
            font_size=dp(25),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            font_name="assets/GTVCS-Medium.ttf"
        )
        self.nrp_container.add_widget(self.nrp_input)

        # Tombol untuk memulai registrasi wajah
        self.register_button = MDRaisedButton(
            text="Daftarkan\nWajah",
            size_hint=(0.25, None),
            font_size=dp(20),
            pos_hint={"center_x": 0.5},
            md_bg_color=(0.631, 0.694, 0.909, 1),
            text_color=(0, 0, 0, 1),
            font_name="assets/GTVCS-Medium.ttf",
            on_release=self.start_face_registration
        )

        self.cancel_registration_button = MDRaisedButton(
            text="Batalkan\nPendaftaran",
            size_hint=(0.25, None),
            font_size=dp(20),
            pos_hint={"center_x": 0.5},
            md_bg_color=(0.913, 0.513, 0.525, 1),
            text_color=(0, 0, 0, 1),
            font_name="assets/GTVCS-Medium.ttf",
            on_release=self.cancel_registration
        )

        # Tambahkan container ke input_layout
        self.cancel_registration_button.opacity = 0
        self.cancel_registration_button.disabled = True
        self.input_layout.add_widget(self.nrp_container)
        self.input_layout.add_widget(self.cancel_registration_button)
        self.input_layout.add_widget(self.register_button)

        self.camera_layout = RelativeLayout(
            size_hint=(1,1),
            pos_hint={"center_x": 0.5, "center_y": 0.66}
        )

        # Container untuk input nama
        self.camera_container = MDBoxLayout(
            size_hint=(None, None),
            size = (dp(850), dp(750)),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            radius=[dp(5), dp(5), dp(5), dp(5)],
            orientation="vertical",
            md_bg_color=(0.219, 0.262, 0.411, 1)
        )

        self.camera_display = Image(size_hint=(None, None), size = (dp(790), dp(750)),
                                    pos_hint={"center_x": 0.5, "center_y": 0.5})
        self.camera_layout.add_widget(self.camera_container)
        self.camera_layout.add_widget(self.camera_display)
        self.camera_layout.add_widget(self.registration_text_container)

        # Tambahkan input_layout ke MainContent
        self.add_widget(self.regist_graphic)
        self.add_widget(self.company_logo)
        self.add_widget(self.camera_layout)
        self.add_widget(self.input_layout)

        self.cap = None
        self.vkeyboard = None
        self.is_registering_face = False
        self.face_encodings_list = []
        self.nrp_input.bind(focus=self.show_keyboard)
        self.update_progress_bar(70)
    
# Fungsi untuk update value dengan animasi halus
    def update_progress_bar(self, value):
        Animation(value=value, duration=0.5).start(self.registration_progressbar)

    def on_enter(self):
        """Aktifkan kamera hanya jika masuk ke Registration"""

        self.cap = CameraSingleton.get_instance(screen_name="Registration")
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        Clock.schedule_interval(self.update_camera_reg, 1.0 / 30.0)

    def on_leave(self):
        """Matikan kamera saat keluar dari Registration"""
        self.stop_camera()
        CameraSingleton.release(screen_name="Registration")

    def stop_camera(self):
        """Hentikan update kamera"""
        Clock.unschedule(self.update_camera_reg)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def update_rect(self, *args):
        self.rect.size = self.size
        self.rect.pos = self.pos



    def start_face_registration(self, *args):
        """Memulai proses registrasi wajah dalam loop update_camera()"""
        nrp = self.nrp_input.text.strip()

        if not nrp:
            warning_nrp_container=MDBoxLayout(
                size_hint=(None, None),
                size=(dp(500), dp(60)),
                pos_hint={"center_x": 0.5, "center_y": 0.5},
                md_bg_color=(0.631, 0.694, 0.909, 1),
                radius=[dp(25), dp(25), dp(25), dp(25)],
                padding=[20, 10, 20, 10]
            )
            warning_nrp_label=MDLabel(text="[size=25]NRP harus diisi![/size]", font_name="assets/GTVCS-Medium.ttf", markup=True, halign="center", pos_hint={"center_x":0.5, "center_y":0.5})
            warning_nrp_container.add_widget(warning_nrp_label)

            warning_nrp_content=MDBoxLayout(orientation='vertical', spacing=10, padding=30, md_bg_color=(0, 0, 0, 0))
            warning_nrp_content.add_widget(Image(source="assets/search_icon.png", size_hint=(1, None), height=250, pos_hint={"center_x":0.5, "center_y":0.4}))
            warning_nrp_content.add_widget(warning_nrp_container)
            self.warning_nrp_dialog = MDDialog(
                type="custom",
                height=50,
                radius=[dp(20), dp(20), dp(20), dp(20)],
                content_cls=warning_nrp_content,
                buttons=[
                    MDRaisedButton(
                        text="[size=18]Tutup[/size]",
                        md_bg_color=(0.631, 0.694, 0.909, 1),
                        text_color=(0.203, 0.2, 0.2, 1),
                        on_release=lambda x: self.warning_nrp_dialog.dismiss()
                    )
                ]  
            )
            self.warning_nrp_dialog.open()
            Clock.schedule_once(lambda x: self.warning_nrp_dialog.dismiss(), 10)
            return

        nrp_exists = any(entry["nrp"] == nrp for entry in database)

        if nrp_exists:
            warning_content = MDBoxLayout(orientation='vertical', spacing=20, padding=20, md_bg_color=(0, 0, 0, 0))
            warning_content.add_widget(Image(
                source="assets/warning_icon.png",
                size_hint=(1, None),
                height=200,
                pos_hint={"center_x":0.5}
            ))

            warning_container=MDBoxLayout(
                size_hint=(None, None),
                size=(dp(500), dp(100)),
                pos_hint={"center_x": 0.5, "center_y": 0.5},
                md_bg_color=(0.631, 0.694, 0.909, 1),
                radius=[dp(25), dp(25), dp(25), dp(25)],
                padding=[20, 10, 20, 10]
            )

            warning_label=MDLabel(text=f"[size=25][b]{nrp}[/b] sudah terdaftar. Apakah Anda ingin mendaftarkan ulang?[/size]", font_name="assets/GTVCS-Medium.ttf", halign="center", pos_hint={"center_x":0.5, "center_y":0.5}, markup=True)
            warning_container.add_widget(warning_label)
            warning_content.add_widget(warning_container)

            self.warning_dialog = MDDialog(
                type="custom",
                content_cls=warning_content,
                radius=[dp(20), dp(20), dp(20), dp(20)],
                buttons=[
                    MDRaisedButton(
                        text="[size=18]Tidak[/size]",
                        font_name="assets/GTVCS-Medium.ttf",
                        text_color=(0, 0, 0, 1),
                        md_bg_color=(0.964, 0.949, 0.878, 1),
                        on_release=lambda x: self.warning_dialog.dismiss()
                    ),
                    MDRaisedButton(
                        text="[size=18]Ya[/size]",
                        font_name="assets/GTVCS-Medium.ttf",
                        md_bg_color=(0.631, 0.694, 0.909, 1),
                        text_color=(0, 0, 0, 1),
                        on_release=lambda x: [self.warning_dialog.dismiss(), self.start_capture_process()]
                    )
                ]
            )
            self.warning_dialog.open()
        else:
            self.start_capture_process()

    def start_capture_process(self):
        self.is_registering_face = True
        self.face_encodings_list = []
        self.registration_text_container.opacity = 1
        self.cancel_registration_button.opacity = 1
        self.cancel_registration_button.disabled = False
        self.update_progress_bar(0)
        self.registration_label.text = "Proses registrasi dimulai, arahkan wajah Anda ke kamera"
        Clock.schedule_interval(self.update_camera_reg, 1.0 / 30.0)
 
 
    def confirm_and_start(self, nrp):
        
        self.nrp_container.opacity = 0
        self.nrp_container.disabled = True
        self.register_button.opacity = 0
        self.register_button.disabled = True
        self.cancel_registration_button.opacity = 1
        self.cancel_registration_button.disabled = False
        self.registration_text_container.opacity = 1

        self.face_encodings_list = []
        self.is_registering_face = True
        self.current_progress = 0
        self.update_progress_bar(0)
        Clock.schedule_interval(self.update_camera_reg, 1.0 / 30.0)  # Update kamera

    def update_camera_reg(self, dt):
        """Mengupdate frame kamera ke widget Image dan menangani registrasi wajah"""
        success, frame = self.cap.read()
        if not success:
            return

        frame = cv2.flip(frame, 1)  # Flip vertikal agar tidak terbalik
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame, stream=False, verbose=False)  # Gunakan frame asli (BGR)
        face_detected = False
        current_time = time.time()
        # Gunakan variabel instance agar tidak selalu None setiap frame
        if not hasattr(self, "locked_face"):
            self.locked_face = None
            self.last_seen_time = 0

        for r in results:
            boxes = r.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = round(box.conf[0].item(), 2)
                cls = int(box.cls[0])

                if conf > confidence and classNames[cls] == "real":
                    # Selalu gambar kotak hijau jika YOLO deteksi wajah real
                    cvzone.cornerRect(frame, (x1, y1, w, h), colorC=(0, 255, 0), colorR=(0, 255, 0))
                    face_detected = True

                    if self.is_registering_face:
                        faces = app.get(img_rgb)
                        # Cari wajah InsightFace yang paling overlap dengan deteksi YOLO
                        best_face = None
                        best_iou = 0
                        for face in faces:
                            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                            xx1 = max(x1, fx1)
                            yy1 = max(y1, fy1)
                            xx2 = min(x2, fx2)
                            yy2 = min(y2, fy2)
                            inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                            box_area = (x2 - x1) * (y2 - y1)
                            face_area = (fx2 - fx1) * (fy2 - fy1)
                            union_area = box_area + face_area - inter_area
                            iou = inter_area / union_area if union_area > 0 else 0
                            if iou > best_iou:
                                best_iou = iou
                                best_face = face
                        if best_face is not None and best_iou > 0.1:
                            face_embedding = best_face.normed_embedding.tolist()
                            if self.locked_face is None:
                                self.locked_face = face_embedding
                                self.last_seen_time = time.time()

                            similarity = np.dot(self.locked_face, face_embedding)
                            if similarity > 0.6:  # Ambang batas kemiripan
                                self.face_encodings_list.append(face_embedding)
                                progress_text = f"Progress data = {int((len(self.face_encodings_list)/15) * 100)}%..."
                                Clock.schedule_once(lambda dt: self.update_registration_text(progress_text))

                                progress_value = (len(self.face_encodings_list) / 15) * 100
                                self.update_progress_bar(progress_value)

                            if len(self.face_encodings_list) >= 15:
                                self.save_face_data()
                                self.stop_face_registration(is_cancelled=False)
                                # Reset lock agar tidak error di frame berikutnya
                                self.locked_face = None
                                self.last_seen_time = 0
                                return

        # Jika wajah terkunci hilang lebih dari 3 detik, reset
        if self.locked_face and not face_detected:
            if (current_time - self.last_seen_time) > 3:
                print("🔓 Wajah hilang! Mencari wajah baru...")
                self.locked_face = None
            else:
                face_detected = True  # Jangan reset jika masih dalam durasi batas

        if face_detected:
            self.last_seen_time = current_time  # Reset timer jika wajah terlihat

        h, w, _ = frame.shape
        x_center, y_center = w // 2, h // 2
        crop_w, crop_h = 960, 1280   # Ukuran yang diinginkan

        x1 = max(0, x_center - crop_w // 2)
        x2 = min(w, x_center + crop_w // 2)
        y1 = max(0, y_center - crop_h // 2)
        y2 = min(h, y_center + crop_h // 2)

        frame = frame[y1:y2, x1:x2]  # Crop gambar

        # Konversi frame ke texture Kivy
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.camera_display.texture = texture  # Setel texture ke kamera

    def update_registration_text(self, text):
        """Memperbarui teks dalam registration_label"""
        self.registration_label.text = text

    def cancel_registration(self, *args):
        self.stop_face_registration(is_cancelled=True)
        
   

    def stop_face_registration(self, is_cancelled=False):
        """Menghentikan proses registrasi wajah dan menampilkan notifikasi sesuai hasil."""
        self.is_registering_face = False
        self.nrp_container.opacity = 1
        self.nrp_container.disabled = False
        self.register_button.opacity = 1
        self.register_button.disabled = False
        self.cancel_registration_button.opacity = 0
        self.cancel_registration_button.disabled = True
        self.registration_text_container.opacity = 0
        self.nrp_input.text = ""

        if is_cancelled:
            self.show_dialog(
                image_src="assets/fail_icon.png",
                bg_color=(0.921, 0.364, 0.380, 1),
                message="[size=25]Registrasi wajah dibatalkan[/size]",
                font_name="assets/GTVCS-Medium.ttf",
                button_cls=MDFlatButton,
                dialog_attr_name="cancel_dialog"
            )
        else:
            self.show_dialog(
                image_src="assets/complete_icon.png",
                bg_color=(0.564, 0.874, 0.654, 1),
                message="[size=25]Registrasi wajah selesai[/size]",
                font_name="assets/GTVCS-Medium.ttf",
                button_cls=MDRaisedButton,
                dialog_attr_name="success_dialog",
                button_color=(0.631, 0.694, 0.909, 1)
            )

    def show_dialog(self, image_src, bg_color, message, font_name, button_cls, dialog_attr_name, button_color=(1, 1, 1, 0)):
        """Menampilkan dialog hasil registrasi wajah (berhasil/gagal)."""

        container = MDBoxLayout(
            size_hint=(None, None),
            size=(dp(500), dp(60)),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            md_bg_color=bg_color,
            radius=[dp(25), dp(25), dp(25), dp(25)],
            padding=[20, 10, 20, 10]
        )

        label = MDLabel(
            text=message,
            font_name=font_name,
            halign="center",
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            markup=True
        )

        container.add_widget(label)

        content = MDBoxLayout(
            orientation='vertical',
            spacing=10,
            padding=30,
            md_bg_color=(0, 0, 0, 0)
        )
        content.add_widget(Image(
            source=image_src,
            size_hint=(1, None),
            height=250,
            pos_hint={"center_x": 0.5, "y": 0.4}
        ))
        content.add_widget(container)

        def close_dialog(instance):
            getattr(self, dialog_attr_name).dismiss()

        button = button_cls(
        text="Tutup",
        text_color=(0.203, 0.2, 0.2, 1),
        on_release=lambda x: getattr(self, dialog_attr_name).dismiss()
        )
        
        if isinstance(button, MDRaisedButton):
            button.md_bg_color = button_color

        dialog = MDDialog(
            type="custom",
            height=50,
            radius=[dp(20), dp(20), dp(20), dp(20)],
            content_cls=content,
            buttons=[button]
        )

        setattr(self, dialog_attr_name, dialog)
        dialog.open()
        Clock.schedule_once(lambda dt: dialog.dismiss(), 10)


    def show_keyboard(self, instance, value):
        """Menampilkan atau menyembunyikan keyboard virtual ketika MDTextField mendapatkan fokus"""
        if value:  # Jika field mendapatkan fokus
            if self.vkeyboard is None:
                self.vkeyboard = VKeyboard()
                self.vkeyboard.size_hint = (1, 0.3)
                self.vkeyboard.bind(on_textinput=self.on_textinput)  # Menangani teks biasa
                self.vkeyboard.bind(on_key_down=self.on_key_down)  # Menangani tombol khusus seperti backspace
                self.add_widget(self.vkeyboard)

        else:  # Jika fokus hilang, sembunyikan keyboard
            if self.vkeyboard:
                self.remove_widget(self.vkeyboard)
                self.vkeyboard = None

    def on_textinput(self, keyboard, text):
        """Menambahkan teks hanya jika valid"""

        # Tentukan MDTextField yang sedang aktif
        active_field = None
        if self.nrp_input.focus:
            active_field = self.nrp_input

        if active_field:
            active_field.text += text  # Tambahkan teks ke field aktif

    def on_key_down(self, keyboard, keycode, text, modifiers):
        """Menangani tombol backspace"""
        # Tentukan MDTextField yang sedang aktif
        active_field = None
        if self.nrp_input.focus:
            active_field = self.nrp_input

        if active_field:
            if keycode == "backspace":
                active_field.text = active_field.text[:-1]  # Hapus karakter terakhir
            elif keycode == "enter":
                active_field.focus = False  # Sembunyikan keyboard saat Enter ditekan
            elif keycode == "escape":
                active_field.focus = False

    def done_notification (self, message):
        """Menampilkan notifikasi pop-up dengan pesan tertentu."""
        self.dialog = MDDialog(
            text=message,
            buttons=[
                MDFlatButton(
                    text="[size=18]OK[/size]",
                    on_release=lambda x: self.dialog.dismiss()
                )
            ]
        )
        self.dialog.open()

    def show_popup_message(self, message, duration=2):
        """Menampilkan popup notifikasi yang akan hilang setelah beberapa detik."""
        if hasattr(self, 'dialog') and self.dialog:  # Cek jika pop-up sudah ada, hapus dulu
            self.dialog.dismiss()
        self.dialog = MDDialog(
            title="Peringatan",
            text=message,
            buttons=[]  # Tidak ada tombol agar otomatis hilang
        )
        self.dialog.open()

        # Jadwalkan agar pop-up tertutup otomatis
        Clock.schedule_once(lambda dt: self.dialog.dismiss(), duration)

    def save_face_data(self):
        nrp = self.nrp_input.text.strip()

        if not nrp:
            print("❗ NRP kosong.")
            return

        if not self.face_encodings_list:
            print("❗ Tidak ada data wajah.")
            return

        mean_encoding = np.mean(self.face_encodings_list, axis=0).tolist()

        # Kirim ke server lebih dulu
        data_to_send = [{"nrp": str(nrp), "encodings": mean_encoding}]
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(Regis_URL, json=data_to_send, headers=headers, timeout=10)
            print(f"📡 Status: {response.status_code}")
            print(f"📡 Response: {response.text}")

            if response.status_code != 200:
                self.show_failed_dialog(f"Gagal kirim ke server. Status: {response.status_code}")
                return

        except Exception as e:
            print("❌ Error saat mengirim ke server:", str(e))
            self.show_failed_dialog("Tidak dapat menghubungi server.\nPastikan Anda terhubung ke internet.")
            return

        # Jika berhasil kirim ke server, baru simpan lokal
        database.append({"nrp": nrp, "encodings": mean_encoding})
        with open(database_json, "w") as file:
            json.dump(database, file, indent=4)
        print(f"✅ Data wajah {nrp} berhasil disimpan di {database_json}")

        database_joblib_data.append({"nrp": nrp, "encodings": mean_encoding})
        joblib.dump(database_joblib_data, joblib_file)
        print(f"✅ Data wajah {nrp} berhasil disimpan di {joblib_file}")

        # Reset registrasi
        self.is_registering_face = False
        self.face_encodings_list = []
        self.stop_face_registration()

    def show_failed_dialog(self, message):
        dialog = MDDialog(
            title="Registrasi Gagal",
            text=message,
            buttons=[MDFlatButton(text="OK", on_release=lambda x: dialog.dismiss())]
        )
        dialog.open()



class ColoredCell(BoxLayout):
    def __init__(self, text, bg_color, **kwargs):
        super().__init__(**kwargs)
        self.size_hint_y = None
        self.height = dp(40)
        self.padding = (5, 5)

        with self.canvas.before:
            Color(1, 0.917, 0.874, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)

        self.bind(size=self.update_rect, pos=self.update_rect)

        label = MDLabel(text=text, halign="center", theme_text_color="Primary")
        self.add_widget(label)

    def update_rect(self, *args):
        self.rect.size = self.size
        self.rect.pos = self.pos

class MainScreen(Screen):
    def __init__(self, screen_manager, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation="horizontal")
        self.cap = None

        self.sidebar = Sidebar(screen_manager=screen_manager)
        self.main_content = MainContent(screen_manager=screen_manager)

        layout.add_widget(self.sidebar)
        layout.add_widget(self.main_content)

        self.add_widget(layout)

    def on_enter(self):
        """Panggil on_enter() di MainContent secara manual"""
        self.main_content.on_enter()

    def on_leave(self):
        """Panggil on_leave() di MainContent secara manual"""
        self.main_content.on_leave()

class RegisterScreen(Screen):
    def __init__(self, screen_manager, **kwargs):
        super().__init__(**kwargs)
        self.screen_manager = screen_manager
        self.sidebar = Sidebar(screen_manager=screen_manager)
        self.main_content = Registration()

        layout = BoxLayout(orientation="horizontal")
        layout.add_widget(self.sidebar)
        layout.add_widget(self.main_content)
        self.add_widget(layout)

class CameraSingleton:
    _instance = None
    _active_screen = None  # Menyimpan layar yang sedang menggunakan kamera

    @staticmethod
    def get_instance(screen_name=None):
        """Pastikan hanya ada satu instance kamera"""
        if CameraSingleton._instance is None or not CameraSingleton._instance.isOpened():
            CameraSingleton._instance = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if CameraSingleton._instance.isOpened():
                CameraSingleton._active_screen = screen_name  # Simpan layar yang sedang pakai kamera

        return CameraSingleton._instance

    @staticmethod
    def release(screen_name=None):
        """Menutup kamera hanya jika layar yang menggunakannya sama"""
        if CameraSingleton._instance is not None:
            if CameraSingleton._active_screen == screen_name:
                CameraSingleton._instance.release()
                CameraSingleton._instance = None
                CameraSingleton._active_screen = None

class myapp(MDApp):
    def build(self):
        sm = ScreenManager()

        main_screen = MainScreen(name="main", screen_manager=sm)
        sm.add_widget(main_screen)

        register_screen = RegisterScreen(name="register", screen_manager=sm)
        sm.add_widget(register_screen)

        sm.current = "main"
        return sm


myapp().run()