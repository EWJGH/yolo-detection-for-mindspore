import hilog from '@ohos.hilog';
import testNapi from 'libentry.so';

//import mindSporeLite from '@ohos.ai.mindSporeLite';
import fs from '@ohos.file.fs';
import { BusinessError } from '@ohos.base';

let model_name = 'sange.ms'
@Entry
@Component
struct Index {
  @State message: string = '开始检测'
  @State bgcolor: string = '#bbbbbb'
  // 控制XComponent组件的创建和销毁
  @State draw: boolean = false

  // xcomponentController: XComponentController = new XComponentController()

  build() {
      Row() {
        Text(this.message)
          .fontSize(40)
          .fontWeight(FontWeight.Bold)
          .margin({ left:50,right: 50 })
          .onClick(() => {
            hilog.info(0x0000, 'testTag', 'start render');

            // 控制XComponent组件的创建和销毁，创建XComponent组件时，会调用libentry.so中的模块注册函数（Init）。
            this.draw = !this.draw;
            if (this.draw) {

              let ret = testNapi.startRender()
              this.message='停止检测'
              this.bgcolor='#eeeeee'
            }
            else {

              let ret = testNapi.stopRender();
              this.message='开始检测'
              this.bgcolor='#bbbbbb'

            }
          })

          XComponent({
            id: 'xcomponent1',
            type: 'surface',
            // controller: this.xcomponentController
            libraryname: 'entry'
          })
            .visibility(this.draw ? Visibility.Visible :Visibility.Hidden)
            .width(640)
            .height(640)
            .onLoad(() => {
              // this.xcomponentController.setXComponentSurfaceSize({surfaceWidth:1920,surfaceHeight:1080});
              // this.surfaceId = this.xcomponentController.getXComponentSurfaceId()
              hilog.info(0x0000, 'testTag', 'onLoad');
            })
            .onDestroy(() => {
              hilog.info(0x0000, 'testTag', 'onDestroy');
            })


      }
      .height('100%')
      .width('100%')
      .backgroundColor(this.bgcolor)
      .justifyContent(FlexAlign.Center)
  }

  aboutToAppear() {
    if (testNapi) {
      console.log("testNapi is not nullptr")
    } else {
      console.log("testNapi is nullptr")
    }
    //testNapi.startRender();
    this.checkMSFile();
  }

  aboutToDisappear() {
    testNapi.stopRender();
  }

  checkMSFile() {
    fs.access('/data/storage/el2/base/haps/entry/files/'+model_name)
      .then((res) => {
        console.info(`=-= access success, res: ${JSON.stringify(res)}}`);
        if (res) {
          console.info(`=-= model has already exist`);
          return;
        } else {
          getContext().resourceManager.getRawFileContent(model_name)
            .then((file) => {
              let newFile: fs.File;

              newFile = fs.openSync('/data/storage/el2/base/haps/entry/files/'+model_name, 0o2 | 0o100);
              console.info(`=-= openSync success, new file fd: ${newFile.fd}`);


              fs.write(newFile.fd, file.buffer).then(() => {
                console.info(`=-= write file success`);
              }).catch((error:BusinessError) => {
                console.info(`=-= write file fail`);
              })
            })
            .catch((err:BusinessError) => {
              console.error(`=-= getRawFileContent fail, err: ${JSON.stringify(err)}`);
            })
        }
      })
      .catch((err:BusinessError) => {
        console.error(`=-= access fail, err: ${JSON.stringify(err)}}`);
      })

  }
}
