# -*- coding: utf-8 -*-
import numpy as np

from OpenGL.GL import *

from .renderbuffer import Renderbuffer, RenderbufferMultisample
from .texture import Texture, TextureMultisample


# class Framebuffer(object):
#     def __init__(self, attachements):
#         self.__id = np.empty(1, dtype=np.uint32)
#         glCreateFramebuffers(len(self.__id), self.__id)
#         for k in list(attachements.keys()):
#             attachement = attachements[k]
#             if isinstance(attachement, Renderbuffer) or isinstance(attachement, RenderbufferMultisample):
#                 glNamedFramebufferRenderbuffer(self.__id, k, GL_RENDERBUFFER, attachement.id)
#             elif isinstance(attachement, Texture) or isinstance(attachement, TextureMultisample):
#                 glNamedFramebufferTexture(self.__id, k, attachement.id, 0)
#             else:
#                 raise ValueError("Unknown frambuffer attachement class: {0}".format(attachement))

#         if glCheckNamedFramebufferStatus(self.__id, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
#             raise RuntimeError("Framebuffer not complete.")
#         self.__attachements = attachements

#     def bind(self):
#         glBindFramebuffer(GL_FRAMEBUFFER, self.__id)

#     def delete(self):
#         glDeleteFramebuffers(1, self.__id)
#         for k in list(self.__attachements.keys()):
#             self.__attachements[k].delete()

#     @property
#     def id(self):
#         return self.__id[0]


class Framebuffer(object):
    def __init__(self, attachements):
        # 1. 生成 framebuffer（注意：取 int）
        self.__id = int(glGenFramebuffers(1))

        # 2. bind
        glBindFramebuffer(GL_FRAMEBUFFER, self.__id)

        # 3. attach
        for k, attachement in attachements.items():
            if isinstance(attachement, (Renderbuffer, RenderbufferMultisample)):
                glFramebufferRenderbuffer(
                    GL_FRAMEBUFFER,
                    k,
                    GL_RENDERBUFFER,
                    attachement.id,
                )
            elif isinstance(attachement, (Texture, TextureMultisample)):
                glFramebufferTexture2D(
                    GL_FRAMEBUFFER,
                    k,
                    GL_TEXTURE_2D,
                    attachement.id,
                    0
                )
            else:
                raise ValueError(f"Unknown framebuffer attachment: {type(attachement)}")

        # 4. check
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Framebuffer not complete: {hex(status)}")

        # 5. unbind
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        self.__attachements = attachements

    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.__id)

    def delete(self):
        glDeleteFramebuffers(1, [self.__id])
        for att in self.__attachements.values():
            att.delete()

    @property
    def id(self):
        return self.__id
