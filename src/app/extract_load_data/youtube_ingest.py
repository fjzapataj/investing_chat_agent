import os
import warnings
import requests
import datetime
import logging
import isodate
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
load_dotenv()

API_KEY = os.getenv("API_KEY_YOUTUBE")
CHANNELS = [
    "Esteban Perez",
    "Bolsas hoy | Invierte y Crece",
    "ARENA ALFA",
    "USACRYPTONOTICIAS",
]
BASE_URL = os.getenv("BASE_URL_YOUTUBE")


class YoutubeIngest:
    """Class to ingest transcriptions and metadata from videos Youtube"""

    def __init__(self):
        self.list_metadata = []
        self.channels = CHANNELS
        self.url = BASE_URL

    def ingest_youtube_videos_from_channels(
        self, channels: list = CHANNELS, daysback: int = 1
    ) -> list:
        list_ingest = []
        for canal in channels:
            logging.info(f"Ingesting data from {canal}")
            channel_id = self.get_channel_id(canal)
            if channel_id:
                videos = self.get_metadata_videos_delta(channel_id, daysback=daysback)
                if len(videos) > 0:
                    for video in videos:
                        video_id = video["videoId"]
                        video["transcript"] = self.get_transcript_by_id(video_id)
                        list_ingest.append(video)
            else:
                logging.error(f"No se pudo obtener el channel_id de {canal}")
        self.list_metadata = list_ingest
        return list_ingest

    @staticmethod
    def get_transcript_by_id(video_id, language="es") -> str:
        """Get the transcript of a video by its id"""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, languages=[language]
            )
            full_transcript = " ".join([entry["text"] for entry in transcript])
            return full_transcript
        except Exception as e:
            logging.error(f"Error al obtener el transcript del video {video_id}: {e}")
            return None

    @staticmethod
    def get_channel_id(channel_name: str) -> str:
        """Get the channel id from the channel name"""
        url = (
            f"{BASE_URL}search?"
            f"key={API_KEY}&"
            f"q={channel_name}&"
            f"part=snippet&type=channel&maxResults=1"
        )
        response = requests.get(url)
        if response.status_code == 200:
            items = response.json().get("items", [])
            if items:
                return items[0]["snippet"]["channelId"]
        return None

    def get_metadata_videos_delta(self, channel_id, daysback=1):
        """Get the metadata of the videos from a channel"""
        delta = datetime.datetime.now(datetime.timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - datetime.timedelta(days=daysback)
        delta_str = delta.strftime("%Y-%m-%dT%H:%M:%SZ")
        video_list = []
        url = (
            f"{BASE_URL}search?"
            f"key={API_KEY}&"
            f"channelId={channel_id}&"
            f"part=snippet&id&"
            f"order=date&"
            f"publishedAfter={delta_str}&"
            f"maxResults=99"
        )

        response = requests.get(url)
        if response.status_code == 200:
            videos = response.json().get("items", [])
            for video in videos:
                video_id = video["id"].get("videoId")
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                video_data = {
                    "videoId": video_id,
                    "title": video["snippet"].get("title"),
                    "publishTime": video["snippet"].get("publishTime"),
                    "videoUrl": video_url,
                    "kind": video["id"].get("kind"),
                    "channelId": video["snippet"].get("channelId"),
                    "channelTitle": video["snippet"].get("channelTitle"),
                }
                duration = self._get_content_details(video_id)
                video_data["duration"] = self.iso8601_to_minutes(duration)
                video_list.append(video_data)
            logging.info(f"Se encontraron {len(video_list)} videos")
        else:
            logging.DEBUG(
                "Error al obtener los datos:", response.status_code, response.text
            )
        return video_list

    @staticmethod
    def _get_content_details(video_id):
        """Get the duration of a video by its id"""
        url = (
            f"{BASE_URL}videos?"
            f"key={API_KEY}&"
            f"id={video_id}&"
            f"part=contentDetails"
        )
        response = requests.get(url)
        if response.status_code == 200:
            items = response.json().get("items", [])
            return items[0]["contentDetails"]["duration"]
        else:
            logging.DEBUG(
                "Error al obtener la duracion del video:",
                response.status_code,
                response.text,
            )
            return None

    @staticmethod
    def iso8601_to_minutes(iso_duration):
        """Convert an ISO8601 duration to minutes"""
        try:
            duracion = isodate.parse_duration(iso_duration)
            total_segundos = duracion.total_seconds()
            minutos = total_segundos // 60
            return minutos
        except Exception as e:
            logging.error(f"Error al convertir la duracion {iso_duration} a minutos")
            return 0
