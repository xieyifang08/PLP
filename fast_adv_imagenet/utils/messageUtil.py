import smtplib
from email.mime.text import MIMEText


class sendMail():
    smtp_adress = 'smtp.qq.com'

    def __init__(self, from_mail, from_passwd, send_list):
        self.from_mail = from_mail
        self.from_passwd = from_passwd
        self.send_list = send_list

    def send(self, send_title, send_content):
        msg = MIMEText(send_content, _charset='utf-8')
        msg['Subject'] = send_title
        msg['From'] = self.from_mail
        msg['To'] = ';'.join(self.send_list)

        mailServer = smtplib.SMTP(self.smtp_adress, 25)
        mailServer.login(self.from_mail, self.from_passwd)
        mailServer.sendmail(self.from_mail, self.send_list, msg.as_string())
        mailServer.quit()


def send_email(message, title='实验结果报告', from_mail='862065233@qq.com', from_passwd='pnuyuzjrwzgnbeee', to_list=['951115439@qq.com']):
    sendMail(from_mail, from_passwd, to_list).send(title, message)
# if __name__ == '__main__':
#     # send_list = ['951115439@qq.com']
#     send_email("1我是内容,第一行\n我是第二行\n我是第三行")

def job_start(message="job start!", title='超算job运行监控', from_mail='862065233@qq.com', from_passwd='pnuyuzjrwzgnbeee', to_list=['951115439@qq.com']):
    send_email(from_mail, from_passwd, to_list).send(title, message)

def job_finish(message="job finish!", title='超算job运行监控', from_mail='862065233@qq.com', from_passwd='pnuyuzjrwzgnbeee', to_list=['951115439@qq.com']):
    send_email(from_mail, from_passwd, to_list).send(title, message)

def job_failed(message="job failed!", title='超算job运行监控', from_mail='862065233@qq.com', from_passwd='pnuyuzjrwzgnbeee', to_list=['951115439@qq.com']):
    send_email(from_mail, from_passwd, to_list).send(title, message)