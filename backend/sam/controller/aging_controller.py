from backend.sam.service.aging_service import AgingService
import logging


class AgingController:

    @staticmethod
    def aging_task(np_arr):
        logging.info("Executing aging task in the sam model controller layer.")
        return AgingService.aging_task(np_arr)

    @staticmethod
    def check_face_alignment(np_arr):
        logging.info(
            "Checking face availability in the sam model controller layer.")
        return AgingService.check_face_alignment(np_arr)
